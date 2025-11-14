import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import gc
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datasets import load_dataset, Dataset
import numpy as np
import tinker
from tinker import types
import wandb
from reward import evaluate
from transformers import AutoTokenizer



BASE_MODEL = "openai/gpt-oss-120b"
DATASET_PATH = Path("dataset.jsonl")
OUTPUT_DIR = "gpt-oss-120b-hs-code-grpo"
RESUME_FROM_CHECKPOINT = None

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

def load_and_prepare_dataset_tinker(renderer, tokenizer) -> Dataset:
    """Load dataset and prepare for Tinker RL training."""
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
    def by_length(example: Dict) -> bool:
        try:
            prompt = renderer.build_generation_prompt(example["prompt"]) 
            text = "".join([chunk.text for chunk in prompt.chunks])
            toks = tokenizer.encode(text)
            return len(toks) <= MAX_PROMPT_TOKENS
        except Exception:
            return False
    dataset = dataset.filter(by_length)
    
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

def create_harmony_reward_func(tokenizer):
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
    os.environ["TINKER_API_KEY"] = os.getenv("TINKER_KEY", "")
    print(f"\n📊 Token Limits Configuration:")
    print(f"   MAX_PROMPT_TOKENS: {MAX_PROMPT_TOKENS}")
    print(f"   MAX_COMPLETION_TOKENS: {MAX_COMPLETION_TOKENS}")
    print(f"   Max total sequence length: {MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS}")
    oom_allocated_gb = 59.20
    estimated_seq_len = estimate_sequence_length_from_oom(oom_allocated_gb, NUM_GENERATIONS)
    print(f"   ⚠️  Previous OOM analysis: ~{estimated_seq_len} tokens estimated from {oom_allocated_gb:.2f} GiB allocation")
    print(f"   Current limits allow up to {MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS} tokens")
    wandb.init(project="hs-code-grpo-tinker", name="gpt-oss-120-grpo-tinker")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()
    renderer = training_client.get_renderer()
    print(f"\n📊 Filtering dataset with current token limits...")
    print(f"   This ensures examples exceeding {MAX_PROMPT_TOKENS} prompt tokens are excluded")
    dataset = load_and_prepare_dataset_tinker(renderer, tokenizer)
    print(f"✓ Dataset loaded: {len(dataset)} examples after filtering")
    reward_func = create_harmony_reward_func(tokenizer)
    global_step = 0
    accum_count = 0
    sampling_client = training_client.save_weights_and_get_sampling_client(name="init")
    for example in dataset:
        messages = example["prompt"]
        user_d = json.loads(example["user_data"])
        targets = json.loads(example["targets"])
        prompt_input = renderer.build_generation_prompt(messages)
        stop_sequences = renderer.get_stop_sequences()
        params = types.SamplingParams(max_tokens=MAX_COMPLETION_TOKENS, temperature=1.1, top_p=0.93, top_k=80, stop=stop_sequences)
        sample_future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=NUM_GENERATIONS)
        sample_result = sample_future.result()
        decoded = [tokenizer.decode(seq.tokens) for seq in sample_result.sequences]
        rewards = reward_func(decoded, [seq.tokens for seq in sample_result.sequences], [user_d] * NUM_GENERATIONS, [targets] * NUM_GENERATIONS)
        mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        data = []
        for i, seq in enumerate(sample_result.sequences):
            adv = float(rewards[i] - mean_reward)
            target_tokens = np.array(seq.tokens, dtype=np.int64)
            logprobs_q = np.array(seq.maybe_logprobs if seq.maybe_logprobs is not None else [0.0] * len(seq.tokens), dtype=np.float32)
            advantages = np.full(len(seq.tokens), adv, dtype=np.float32)
            datum = types.Datum(model_input=prompt_input, loss_fn_inputs={"target_tokens": target_tokens, "logprobs": logprobs_q, "advantages": advantages})
            data.append(datum)
        fwdbwd_future = training_client.forward_backward(data, "importance_sampling")
        fwdbwd_result = fwdbwd_future.result()
        accum_count += 1
        if accum_count >= GRADIENT_ACCUMULATION_STEPS:
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))
            optim_result = optim_future.result()
            accum_count = 0
            global_step += 1
            wandb.log({"step": global_step, "mean_reward": mean_reward})
            if global_step % SAVE_STEPS == 0:
                save_result = training_client.save_state(name=f"step_{global_step}")
                print(f"✓ Saved checkpoint: {save_result.path}")
                sampling_client = training_client.save_weights_and_get_sampling_client(name=f"sampler_{global_step}")
    final_save = training_client.save_state(name=f"final_{global_step}")
    wandb.finish()
    print(f"\n Training complete! Saved to {final_save.path}")

if __name__ == "__main__":
    main()