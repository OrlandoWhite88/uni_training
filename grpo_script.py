import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import gc
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TrainerCallback
from peft import PeftModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
from reward import evaluate


MODEL_ID = "openai/gpt-oss-20b"
SFT_ADAPTER_PATH = "uni_sft_model"
DATASET_PATH = Path("dataset.jsonl")
OUTPUT_DIR = "gpt-oss-20b-hs-code-grpo"
RESUME_FROM_CHECKPOINT = None  # Set to None to start fresh

# Token limits - reduced to prevent OOM with eager attention (O(n²) memory)
MAX_PROMPT_TOKENS = 8000  # Reduced to prevent OOM
MAX_COMPLETION_TOKENS = 2048  # Keep at 2048 as requested

WARMUP_STEPS = 100
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 4
SAVE_STEPS = 5


def prepare_grpo_dataset(example: Dict) -> Dict:
    """Convert Harmony messages to GRPO format with raw message lists."""
    messages = example["messages"]
    
    # Parse JSON payloads
    user_data = json.loads(messages[1]["content"])
    targets = json.loads(messages[2]["content"])
    
    # Raw conversational prompt (GRPO will apply chat template)
    prompt_messages = [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user", "content": messages[1]["content"]}
    ]
    
    return {
        "prompt": prompt_messages,  # List of dicts, not string!
        "user_data": json.dumps(user_data),
        "targets": json.dumps(targets)
    }

def filter_by_task_type(example: Dict) -> bool:
    """Filter examples to only include supported task types."""
    try:
        user_data = json.loads(example["user_data"])
        task_type = user_data.get("task")
        supported_tasks = {"select_chapters", "select_candidates", "score_candidate"}
        return task_type in supported_tasks
    except:
        return False

def filter_by_length(example: Dict, tokenizer: AutoTokenizer, max_tokens: int) -> bool:
    """Filter examples exceeding token limit."""
    tokens = tokenizer.apply_chat_template(
        example["prompt"],
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens=False
    )
    return len(tokens) <= max_tokens

def load_and_prepare_dataset(tokenizer: AutoTokenizer) -> Dataset:
    """Load dataset and prepare for GRPO training."""
    dataset_path = DATASET_PATH.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset not found at {dataset_path}")
    
    raw_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train"
    )
    
    dataset = raw_dataset.map(
        prepare_grpo_dataset,
        remove_columns=raw_dataset.column_names
    )
    
    # Filter by supported task types first
    dataset = dataset.filter(filter_by_task_type)
    print(f"✓ Filtered by task type: {len(dataset)} examples remaining")
    
    # Then filter by length
    dataset = dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "max_tokens": MAX_PROMPT_TOKENS}
    )
    
    if len(dataset) == 0:
        raise ValueError("No examples loaded after filtering by token length.")
    
    return dataset

def extract_completion_text(completion: Any) -> str:
    """Extract text content from completion (handles string, list, dict)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Extract content from list of message dicts
        texts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
        return "\n".join(texts)
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(completion)

def extract_last_json(text: str) -> Optional[str]:
    """Extract the last JSON object from text."""
    if not isinstance(text, str):
        return None
    # Find the last opening brace
    last_open = text.rfind('{')
    if last_open == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    for i in range(last_open, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[last_open:i+1]
    
    return None

def create_harmony_reward_func(tokenizer: AutoTokenizer):
    """Create a reward function with tokenizer bound via closure."""
    def harmony_reward_func(
        completions: List[str],  # Decoded without special tokens
        completion_ids: List[List[int]],  # Raw token IDs (not used)
        user_data: List[Dict],
        targets: List[Dict],
        **kwargs
    ) -> List[float]:
        """Extract last JSON from completion and evaluate."""
        rewards = []
        
        # Print summary header
        print(f"\n{'='*80}")
        print(f"REWARD CALCULATION - Processing {len(completions)} completions")
        print(f"{'='*80}")
        
        for idx, (completion, user_d, target) in enumerate(zip(completions, user_data, targets)):
            try:
                # Extract text content from completion (handles string, list, dict)
                completion_text = extract_completion_text(completion)
                
                # Print raw completion for first few examples
                if idx < 3:
                    print(f"\n{'='*80}")
                    print(f"RAW COMPLETION #{idx+1} (type: {type(completion).__name__}, {len(completion_text)} chars):")
                    print(f"{'='*80}")
                    print(completion_text)
                    print(f"{'='*80}")
                
                if isinstance(user_d, str):
                    user_d = json.loads(user_d)
                if isinstance(target, str):
                    target = json.loads(target)
                
                # Extract last JSON from completion text
                json_text = extract_last_json(completion_text)
                
                # Print extraction details for first few examples
                if idx < 3:
                    print(f"\nCompletion #{idx+1} Details:")
                    print(f"  Extracted JSON: {json_text[:500] if json_text else 'None'}...")
                    print(f"  JSON length: {len(json_text) if json_text else 0}")
                    print(f"  User data task: {user_d.get('task', 'unknown')}")
                
                if json_text is None:
                    if idx < 3:
                        print(f"⚠️  WARNING: No JSON found in completion!")
                        print(f"   Completion text length: {len(completion_text)}")
                    rewards.append(0.0)
                    continue
                
                # Use your existing reward module
                result = evaluate(
                    user_data=user_d,
                    answer=json_text.strip(),
                    targets=target
                )
                
                score = result.get("score", 0.0) if result.get("is_score_valid") else 0.0
                
                # Print reward details for first few examples
                if idx < 3:
                    print(f"\n  Reward calculation:")
                    print(f"    Score: {score}")
                    print(f"    Is valid: {result.get('is_score_valid', False)}")
                    print(f"    Reason: {result.get('reason', 'N/A')}")
                    print(f"{'='*80}")
                
                rewards.append(score)
                
            except Exception as e:
                if idx < 3:  # Print errors for first few examples
                    print(f"\n{'='*80}")
                    print(f"❌ REWARD ERROR for completion #{idx+1}:")
                    print(f"{'='*80}")
                    print(f"Error: {e}")
                    try:
                        error_text = extract_completion_text(completion)
                        print(f"Raw completion text: {error_text[:500]}...")
                    except:
                        print(f"Raw completion (could not extract text): {str(completion)[:500]}...")
                    import traceback
                    traceback.print_exc()
                    print(f"{'='*80}\n")
                rewards.append(0.0)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"REWARD SUMMARY:")
        print(f"  Total completions: {len(rewards)}")
        print(f"  Rewards: {rewards}")
        print(f"  Mean reward: {sum(rewards) / len(rewards) if rewards else 0:.4f}")
        print(f"  Zero rewards: {sum(1 for r in rewards if r == 0.0)}")
        print(f"{'='*80}\n")
        
        return rewards
    
    return harmony_reward_func

class MemoryClearCallback(TrainerCallback):
    """Callback to clear CUDA cache and run garbage collection every N steps."""
    
    def __init__(self, clear_every_n_steps: int = 10):
        self.clear_every_n_steps = clear_every_n_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        """Clear memory every N steps."""
        if state.global_step % self.clear_every_n_steps == 0:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            if state.global_step % (self.clear_every_n_steps * 10) == 0:
                # Print memory stats every 100 steps for monitoring
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    print(f"🧹 Memory cleared at step {state.global_step} | "
                          f"GPU Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        return control



def estimate_sequence_length_from_oom(allocated_gb: float, num_generations: int = 4) -> int:
    """
    Estimate the sequence length that caused OOM with eager attention.
    
    With eager attention (O(n²)), memory scales quadratically with sequence length.
    For GPT-OSS-20B: ~40 layers, ~16-32 attention heads, bfloat16 (2 bytes)
    
    Rough estimate: Memory ≈ num_layers * num_heads * num_generations * seq_len² * 2 bytes
    Solving for seq_len given the OOM allocation.
    """
    # GPT-OSS-20B architecture estimates
    num_layers = 40
    num_heads = 16  # Conservative estimate
    bytes_per_element = 2  # bfloat16
    
    # Convert GB to bytes
    allocated_bytes = allocated_gb * (1024 ** 3)
    
    # Rough formula: memory ≈ layers * heads * generations * seq_len² * bytes
    # This is simplified - actual memory includes K/V cache, hidden states, etc.
    # So we use a factor to account for overhead
    overhead_factor = 3  # Account for K/V cache, hidden states, gradients, etc.
    
    # Solve: allocated_bytes ≈ overhead_factor * num_layers * num_heads * num_generations * seq_len² * bytes_per_element
    seq_len_squared = allocated_bytes / (overhead_factor * num_layers * num_heads * num_generations * bytes_per_element)
    estimated_seq_len = int(math.sqrt(seq_len_squared))
    
    return estimated_seq_len

# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    # Print token limits being used (important for debugging OOM)
    print(f"\n📊 Token Limits Configuration:")
    print(f"   MAX_PROMPT_TOKENS: {MAX_PROMPT_TOKENS}")
    print(f"   MAX_COMPLETION_TOKENS: {MAX_COMPLETION_TOKENS}")
    print(f"   Max total sequence length: {MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS}")
    
    # Estimate sequence length from previous OOM (if known)
    # Based on: "Tried to allocate 59.20 GiB"
    oom_allocated_gb = 59.20
    estimated_seq_len = estimate_sequence_length_from_oom(oom_allocated_gb, NUM_GENERATIONS)
    print(f"   ⚠️  Previous OOM analysis: ~{estimated_seq_len} tokens estimated from {oom_allocated_gb:.2f} GiB allocation")
    print(f"   Current limits allow up to {MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS} tokens (reduced to prevent OOM)")
    print()
    
    # Initialize wandb
    wandb.init(project="hs-code-grpo", name="gpt-oss-grpo-sft-continue")
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 2. Load model with MXFP4 quantization (GPT-OSS specific)
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,
        device_map="auto",
    )
    
    # Check if we're resuming from a checkpoint
    checkpoint_path = Path(OUTPUT_DIR) / RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT else None
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        # Verify checkpoint contains training state
        trainer_state_file = checkpoint_path / "trainer_state.json"
        if trainer_state_file.exists():
            import json
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            global_step = trainer_state.get("global_step", 0)
            print(f"   Found training state at step {global_step}")
            print(f"   Learning rate scheduler will continue from step {global_step}")
        
        # Load model from checkpoint (should include PEFT adapters)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=True)
        print(f"✓ Loaded model from checkpoint {RESUME_FROM_CHECKPOINT}")
        
        # Verify adapter is enabled and trainable
        model.train()  # Ensure model is in training mode
        if hasattr(model, 'peft_config'):
            print(f"✓ Adapter config: {model.peft_config}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Trainable parameters: {trainable_params:,}")
        base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"✓ Base model trainable params (should be 0): {base_trainable}")
    else:
        print(f"🆕 Starting fresh training")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        # Load SFT adapter as starting point
        model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
        print(f"✓ Loaded SFT adapter from {SFT_ADAPTER_PATH}")
        
        # Verify adapter is enabled and trainable
        model.train()  # Ensure model is in training mode
        if hasattr(model, 'peft_config'):
            print(f"✓ Adapter config: {model.peft_config}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Trainable parameters: {trainable_params:,}")
        base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"✓ Base model trainable params (should be 0): {base_trainable}")
    
    # 3. Prepare dataset - ALWAYS re-filter with current token limits
    # This ensures that if token limits are changed between runs, the dataset
    # is filtered accordingly, preventing OOM from examples that exceed limits
    print(f"\n📊 Filtering dataset with current token limits...")
    print(f"   This ensures examples exceeding {MAX_PROMPT_TOKENS} prompt tokens are excluded")
    dataset = load_and_prepare_dataset(tokenizer)
    print(f"✓ Dataset loaded: {len(dataset)} examples after filtering")
    
    # 4. GRPO Training Configuration
    training_args = GRPOConfig(
        # I/O
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,  # Don't overwrite when resuming
        push_to_hub=False,  # Save locally only
        
        # Training
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # LR schedule
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        
        # Generation (12k prompt, 2k completion)
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=MAX_COMPLETION_TOKENS,
        num_generations=NUM_GENERATIONS,
        temperature=1.1,  # Diversity in sampling (temperature > 0 enables sampling)
        top_p=0.93,  # Slightly loosen nucleus sampling for more variety
        top_k=80,  # Modest cap keeps tail in check while broadening choices
        generation_kwargs={
            "repetition_penalty": 1.05,
            "typical_p": 0.9,
        },
        
        
        # RL parameters
        beta=0,  # KL penalty
        epsilon=0.2,  # UP from 0.2
        importance_sampling_level="sequence",
        scale_rewards=False,  # Dr. GRPO paper recommends no scaling
        loss_type="dapo",  # Avoid length bias
        mask_truncated_completions=True,
        
        # Chat template kwargs
        chat_template_kwargs={"reasoning_effort": "high"},
        
        # Logging & inspection
        logging_steps=1,
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=10,
        
        # Saving
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        
        # Memory
        gradient_checkpointing=True,
        remove_unused_columns=False,
        
        # Data preprocessing
        shuffle_dataset=True,
    )
    
    # 5. Initialize GRPOTrainer
    reward_func = create_harmony_reward_func(tokenizer)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    # Add memory clearing callback
    memory_callback = MemoryClearCallback(clear_every_n_steps=10)
    trainer.add_callback(memory_callback)
    print("✓ Added memory clearing callback (clears every 10 steps)")
    
    # 6. Train
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n🔄 Resuming GRPO training from {RESUME_FROM_CHECKPOINT}...")
        print(f"   Checkpoint path: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=str(checkpoint_path))
    else:
        print("\n🚀 Starting GRPO training from scratch...")
        trainer.train()
    
    # 7. Save locally
    trainer.save_model()
    wandb.finish()
    print(f"\n Training complete! Model saved locally to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()