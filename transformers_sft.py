import json
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# -------------------- Configuration ------------------
MODEL_NAME = "openai/gpt-oss-20b"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "uni_grpo_model"

# Wandb Configuration
WANDB_PROJECT = "gpt-oss-sft"
WANDB_RUN_NAME = "harmony-high-reasoning-v1"
WANDB_TAGS = ["harmony", "high-reasoning", "classification"]

# PEFT Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_DROPOUT = 0.1

# Training parameters - MEMORY EFFICIENT
MAX_LENGTH = 14000
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
NUM_EPOCHS = 1
MAX_STEPS = 2500
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100

# -------------------- Dataset Processing ------------------
def extract_training_pair(example: dict, tokenizer) -> dict:
    """
    Extract prompt and completion for SFT with CORRECT Harmony channel handling.
    Enables high reasoning mode via chat template parameter.

    Prompt: All messages up to (but not including) assistant message
    Completion: Channels built from thinking/content fields ONLY
    """
    messages = example["messages"]

    # Find the last assistant message index
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not assistant_indices:
        raise ValueError("No assistant message found in conversation")

    last_assistant_idx = assistant_indices[-1]

    # Get all messages before the assistant response (for prompt)
    prompt_messages = messages[:last_assistant_idx]

    # Build prompt with HIGH REASONING mode
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=False,
        reasoning_effort="high"  # This sets Reasoning: high in the system message
    ).rstrip() + "<|start|>assistant<|message|>"

    # Build completion from assistant's thinking/content fields
    assistant_msg = messages[last_assistant_idx]

    # Verify required fields exist
    if "thinking" not in assistant_msg or "content" not in assistant_msg:
        raise ValueError(f"Assistant message missing 'thinking' or 'content' field: {assistant_msg}")

    thinking = assistant_msg["thinking"]
    content = assistant_msg["content"]

    # Build channels in correct Harmony format
    completion_parts = []
    if thinking:
        completion_parts.append(f"<|channel|>analysis<|message|>{thinking}<|end|>")
    if content:
        completion_parts.append(f"<|channel|>final<|message|>{content}<|end|>")

    completion_text = "".join(completion_parts)

    return {
        "prompt": prompt_text,
        "completion": completion_text,
    }

def prepare_sft_dataset(examples: dict, tokenizer) -> dict:
    """Format dataset for SFT training using proper prompt-completion format"""
    result = {
        "prompt": [],
        "completion": []
    }

    for messages in examples["messages"]:
        pair = extract_training_pair({"messages": messages}, tokenizer)
        result["prompt"].append(pair["prompt"])
        result["completion"].append(pair["completion"])

    return result

def load_and_prepare_dataset(dataset_path: str, tokenizer) -> tuple:
    """Load and prepare dataset for SFT training - MEMORY EFFICIENT"""
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    processed_dataset = raw_dataset.map(
        lambda x: prepare_sft_dataset(x, tokenizer),
        batched=True,
        batch_size=10,
        remove_columns=raw_dataset.column_names,
        desc="Formatting for SFT",
        num_proc=1
    )

    return processed_dataset.train_test_split(test_size=0.05, seed=3407)

# -------------------- Model Setup ------------------
def setup_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with MXFP4 quantization - MEMORY EFFICIENT"""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=MAX_LENGTH
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
        use_cache=False,
    )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
        bias="none",
        use_rslora=True,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

# -------------------- Main Training ------------------
def main():
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        tags=WANDB_TAGS,
        config={
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "warmup_steps": WARMUP_STEPS,
            "num_epochs": NUM_EPOCHS,
            "max_steps": MAX_STEPS,
        }
    )

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

    # Load dataset
    print("Loading and preparing dataset...")
    train_val_datasets = load_and_prepare_dataset(DATASET_PATH, tokenizer)
    train_dataset = train_val_datasets["train"]
    eval_dataset = train_val_datasets["test"]

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # DEBUG: Print first 3 raw completions without truncation to verify format
    print("\n" + "="*80)
    print("DEBUG: First 3 raw completions (verify format)")
    print("="*80)
    for i in range(min(3, len(train_dataset))):
        example = train_dataset[i]
        full_text = example['prompt'] + example['completion']
        print(f"\n--- Example {i+1} ---")
        print(full_text)
        print("-" * 80)

    # Configure training WITH WANDB
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        report_to=["wandb"],  # Enable wandb logging
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        dataset_text_field=None,
        packing=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        run_name=WANDB_RUN_NAME,  # Optional: sync run name with wandb
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training with wandb logging...")
    print("With completion_only_loss=True, model learns to generate completions given prompts")
    print("but gradients only flow from completion tokens.\n")
    trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Finish wandb run
    wandb.finish()

    print(f"SFT training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()