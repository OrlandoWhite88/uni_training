from __future__ import annotations
from unsloth import FastLanguageModel

import json
import os
from typing import List

import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
import wandb

MAX_SEQUENCE_LENGTH = 14000
VALIDATION_SPLIT_RATIO = 0.05
RANDOM_SEED = 3407
DATASET_PATH = "dataset.jsonl"

MODEL_IDENTIFIER = "openai/gpt-oss-120b"
DATA_TYPE = None

GLOBAL_RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN_PROCESS = GLOBAL_RANK == 0
DEVICE_MAP: dict | None = None

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE_MAP = {"": f"cuda:{LOCAL_RANK}"}

# Wandb configuration - enabled by default
WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "0").lower() in {"1", "true", "yes"}
WAND_PROJECT = os.environ.get("WANDB_PROJECT", "gpt-oss-sft")
WAND_ENTITY = os.environ.get("WANDB_ENTITY")
WAND_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "gpt-oss-finetune")
IS_WANDB_ENABLED = not WANDB_DISABLED

if IS_WANDB_ENABLED:
    os.environ["WANDB_PROJECT"] = WAND_PROJECT
    if WAND_ENTITY:
        os.environ["WANDB_ENTITY"] = WAND_ENTITY
    os.environ.setdefault("WANDB_MODE", "online")

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_IDENTIFIER,
    dtype=DATA_TYPE,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
    device_map=DEVICE_MAP,
)

GRADIENT_CHECKPOINTING_MODE: str | bool | None = os.environ.get("GRADIENT_CHECKPOINTING_MODE", "unsloth")
if WORLD_SIZE > 1 and GRADIENT_CHECKPOINTING_MODE == "unsloth":
    GRADIENT_CHECKPOINTING_MODE = False
    if IS_MAIN_PROCESS:
        print("Disabling gradient checkpointing for multi-GPU DDP run to avoid reentrant backward conflicts.")

fine_tuned_model = FastLanguageModel.get_peft_model(
    model=base_model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=GRADIENT_CHECKPOINTING_MODE,
    random_state=RANDOM_SEED,
    use_rslora=True,
)

# Initialize wandb after model loading but before dataset loading
# This ensures the API key prompt appears early, and we can log model info
if IS_WANDB_ENABLED and IS_MAIN_PROCESS:
    wandb.init(
        project=WAND_PROJECT,
        entity=WAND_ENTITY,
        name=WAND_RUN_NAME,
        config={
            "model": MODEL_IDENTIFIER,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "num_train_epochs": 1,
            "max_steps": 2500,
        }
    )

def load_jsonl_dataset(dataset_path: str) -> Dataset:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    return dataset

def format_conversations_for_training(batch: dict) -> dict:
    conversations = batch["messages"]
    formatted_texts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort="high",
        )
        for conversation in conversations
    ]
    return {"text": formatted_texts}

def split_dataset_into_train_and_eval(dataset_path: str) -> tuple[Dataset, Dataset]:
    raw_dataset = load_jsonl_dataset(dataset_path)
    formatted_dataset = raw_dataset.map(
        format_conversations_for_training,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    train_val_split = formatted_dataset.train_test_split(
        test_size=VALIDATION_SPLIT_RATIO,
        seed=RANDOM_SEED,
    )
    training_dataset = train_val_split["train"]
    evaluation_dataset = train_val_split["test"]
    return training_dataset, evaluation_dataset

training_dataset, evaluation_dataset = split_dataset_into_train_and_eval(DATASET_PATH)
if IS_MAIN_PROCESS:
    print("Example formatted prompt:")
    print(training_dataset[0]["text"][:500])

evaluation_dataset = None

trainer = SFTTrainer(
    model=fine_tuned_model,
    tokenizer=tokenizer,
    train_dataset=training_dataset,
    eval_dataset=None,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=1,
        max_steps=2500,
        learning_rate=2e-4,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=RANDOM_SEED,
        output_dir="outputs",
        report_to=["wandb"] if IS_WANDB_ENABLED and IS_MAIN_PROCESS else [],
        run_name=WAND_RUN_NAME if IS_WANDB_ENABLED and IS_MAIN_PROCESS else None,
        eval_strategy="no",
        save_steps=5,
    ),
)

trainer.train()

# Finish wandb run if enabled
if hasattr(trainer, "accelerator"):
    trainer.accelerator.wait_for_everyone()
if IS_WANDB_ENABLED and IS_MAIN_PROCESS:
    wandb.finish()

if IS_MAIN_PROCESS:
    fine_tuned_model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")
