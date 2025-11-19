"""
hs_rl_tinker.py

Reinforcement learning on HS classification with Tinker.

- Uses your existing Harmony-style dataset.jsonl and reward.evaluate(...)
- Starts from a LoRA head on top of a Tinker base model (e.g. openai/gpt-oss-20b)
- Implements a GRPO-style loop using Tinker's "importance_sampling" loss.

Requirements:
  pip install "git+https://github.com/thinking-machines-lab/tinker.git"
  pip install "git+https://github.com/thinking-machines-lab/tinker-cookbook.git"

Env:
  export TINKER_API_KEY=...
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import torch
import tinker
import wandb
from datasets import Dataset
from tinker import types
from tinker.types.tensor_data import TensorData

from reward import evaluate  # your existing reward.py


# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

MODEL_ID = "openai/gpt-oss-120b"  # must be one of Tinker's supported models
DATASET_PATH = Path("dataset.jsonl")

# Tinker / LoRA
LORA_RANK = 32
BASE_URL: Optional[str] = None  # e.g. "https://api.tinker.xyz" or None for default
RESUME_STATE_PATH: Optional[str] = None  # tinker://.../state/... if resuming

# RL / sampling hyperparams
BATCH_SIZE = 64             # number of prompts per RL step
GROUP_SIZE = 4              # number of rollouts per prompt (was NUM_GENERATIONS)
MAX_PROMPT_TOKENS = 32000
MAX_COMPLETION_TOKENS = 2048
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1

TEMPERATURE = 1.1
TOP_P = 0.93

SAVE_EVERY = 10             # save Tinker state every N steps

SUPPORTED_TASKS = {"select_chapters", "select_candidates", "score_candidate"}


# ------------------------------------------------------------------------------------
# DATASET PREPARATION  (same semantics as your GRPO script)
# ------------------------------------------------------------------------------------

def prepare_grpo_dataset(example: Dict) -> Dict:
    """
    Convert Harmony messages to a simpler format we can use for RL:

    - 'prompt' is a list of system+user messages.
    - 'user_data' and 'targets' are JSON-encoded strings.

    messages layout is assumed:
      [0] system
      [1] user (JSON as string)
      [2] assistant (targets JSON as string)
    """
    messages = example["messages"]

    user_data = json.loads(messages[1]["content"])
    targets = json.loads(messages[2]["content"])

    prompt_messages = [
        {"role": "system", "content": messages[0]["content"]},
        {"role": "user", "content": messages[1]["content"]},
    ]

    return {
        "prompt": prompt_messages,
        "user_data": json.dumps(user_data),
        "targets": json.dumps(targets),
    }


def filter_by_task_type(example: Dict) -> bool:
    """
    Keep only examples whose user_data['task'] is in SUPPORTED_TASKS.

    Malformed JSON or missing 'task' -> drop.
    """
    try:
        user_data = json.loads(example["user_data"])
    except (TypeError, json.JSONDecodeError):
        return False

    task_type = user_data.get("task")
    return task_type in SUPPORTED_TASKS


def filter_by_length(example: Dict, tokenizer, max_tokens: int) -> bool:
    """
    Filter out examples whose tokenized prompt exceeds max_tokens.
    Uses the HF chat template for gpt-oss.
    """
    tokens = tokenizer.apply_chat_template(
        example["prompt"],
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens=False,
    )
    return len(tokens) <= max_tokens


def load_and_prepare_dataset(tokenizer) -> Dataset:
    """
    Load dataset, convert to RL format, and filter by task type + length.

    Semantics:
    - Raises FileNotFoundError if DATASET_PATH is missing.
    - Raises ValueError if no examples remain after filtering.
    """
    dataset_path = DATASET_PATH.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Local dataset not found at {dataset_path}")

    raw_dataset = datasets.load_dataset(
        "json", data_files=str(dataset_path), split="train"
    )

    dataset = raw_dataset.map(
        prepare_grpo_dataset,
        remove_columns=raw_dataset.column_names,
    )

    dataset = dataset.filter(filter_by_task_type)
    print(f"Filtered by task type: {len(dataset)} examples")

    dataset = dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "max_tokens": MAX_PROMPT_TOKENS},
    )
    print(f"Filtered by length: {len(dataset)} examples")

    if len(dataset) == 0:
        raise ValueError("No examples remaining after filtering by task and length.")

    return dataset


# ------------------------------------------------------------------------------------
# COMPLETION → REWARD
# ------------------------------------------------------------------------------------

def extract_last_json(text: Any) -> Optional[str]:
    """
    Extract the last JSON object (by braces) from a text string.

    - Non-string input -> None.
    - No '{' found or unmatched braces -> None.
    - Otherwise returns the substring from last '{' to its matching '}'.
    """
    if not isinstance(text, str):
        return None

    last_open = text.rfind("{")
    if last_open == -1:
        return None

    brace_count = 0
    for i in range(last_open, len(text)):
        char = text[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[last_open : i + 1]

    return None


def compute_harmony_reward_from_tokens(
    sampled_tokens: List[int],
    user_data: Dict[str, Any],
    targets: Dict[str, Any],
    tokenizer,
) -> float:
    """
    Decode sampled tokens, extract last JSON, and call your reward.evaluate(...).

    Behaviour:
    - If no JSON is found -> 0.0
    - If evaluate(...) returns is_score_valid=False -> 0.0
    - On any error -> 0.0
    """
    try:
        completion_text = tokenizer.decode(sampled_tokens)
        json_text = extract_last_json(completion_text)
        if json_text is None:
            return 0.0

        result = evaluate(
            user_data=user_data,
            answer=json_text.strip(),
            targets=targets,
        )

        if not result.get("is_score_valid"):
            return 0.0

        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


# ------------------------------------------------------------------------------------
# TINKER SETUP
# ------------------------------------------------------------------------------------

def create_training_client() -> tinker.TrainingClient:
    service_client = tinker.ServiceClient(base_url=BASE_URL)

    if RESUME_STATE_PATH:
        print(f"Resuming training from state: {RESUME_STATE_PATH}")
        training_client = service_client.create_training_client_from_state(
            RESUME_STATE_PATH
        )
    else:
        print(f"Starting new LoRA training run from base model: {MODEL_ID}")
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_ID,
            rank=LORA_RANK,
        )

    return training_client


# ------------------------------------------------------------------------------------
# RL TRAINING LOOP
# ------------------------------------------------------------------------------------

def train_with_tinker() -> None:
    wandb.init(project="hs-code-grpo", name="gpt-oss-20b-tinker-rl")

    # 1) Setup Tinker
    training_client = create_training_client()
    service_client = tinker.ServiceClient(base_url=BASE_URL)

    # 2) Tokenizer (comes from Tinker so it matches the model)
    tokenizer = training_client.get_tokenizer()

    # 3) Dataset
    dataset = load_and_prepare_dataset(tokenizer)
    n_examples = len(dataset)

    n_batches = math.ceil(n_examples / BATCH_SIZE)
    print(f"Dataset size: {n_examples}, batches/epoch: {n_batches}")

    # 4) Sampling + optimizer config
    sampling_params = types.SamplingParams(
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        # stop can be left None; EOS or max_tokens will terminate
    )
    adam_params = types.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        # shuffle each epoch
        dataset_epoch = dataset.shuffle(seed=epoch)

        for batch_idx in range(n_batches):
            t_start = time.time()

            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, n_examples)
            batch_rows = dataset_epoch.select(range(start, end))

            # ------------------------------------------------------------
            # 1) Create sampling client from current weights
            # ------------------------------------------------------------
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"step_{global_step:06d}"
            )

            # ------------------------------------------------------------
            # 2) Generate rollouts (GROUP_SIZE completions per prompt)
            # ------------------------------------------------------------
            batch_futures: List[List["types.SampleFuture"]] = []
            batch_prompt_tokens: List[List[int]] = []
            batch_user_data: List[Dict[str, Any]] = []
            batch_targets: List[Dict[str, Any]] = []

            for row in batch_rows:
                user_data = json.loads(row["user_data"])
                targets = json.loads(row["targets"])
                prompt_messages = row["prompt"]

                prompt_tokens = tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    add_special_tokens=False,
                )

                model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

                sample_futures = []
                for _ in range(GROUP_SIZE):
                    fut = sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    sample_futures.append(fut)

                batch_futures.append(sample_futures)
                batch_prompt_tokens.append(prompt_tokens)
                batch_user_data.append(user_data)
                batch_targets.append(targets)

            # ------------------------------------------------------------
            # 3) Compute rewards + build Tinker Datums
            # ------------------------------------------------------------
            training_datums: List[types.Datum] = []
            batch_mean_rewards: List[float] = []

            for sample_futures, prompt_tokens, user_data, targets in zip(
                batch_futures, batch_prompt_tokens, batch_user_data, batch_targets
            ):
                group_rewards: List[float] = []
                group_tokens: List[List[int]] = []
                group_logprobs: List[List[float]] = []

                # gather completions for this prompt
                for fut in sample_futures:
                    sample_result = fut.result()
                    seq = sample_result.sequences[0]
                    sampled_tokens = seq.tokens
                    sampled_logprobs = seq.logprobs
                    assert sampled_logprobs is not None, "logprobs must be returned"

                    # full token sequence = prompt + completion
                    all_tokens = prompt_tokens + sampled_tokens

                    group_tokens.append(all_tokens)
                    group_logprobs.append(sampled_logprobs)

                    # reward from your Harmony evaluator
                    reward = compute_harmony_reward_from_tokens(
                        sampled_tokens=sampled_tokens,
                        user_data=user_data,
                        targets=targets,
                        tokenizer=tokenizer,
                    )
                    group_rewards.append(reward)

                if len(group_rewards) == 0:
                    continue

                mean_reward = sum(group_rewards) / len(group_rewards)
                batch_mean_rewards.append(mean_reward)

                # GRPO-style advantages (reward - group mean)
                advantages = [
                    r - mean_reward for r in group_rewards
                ]

                # if all advantages are exactly zero, skip (no signal)
                if all(a == 0.0 for a in advantages):
                    continue

                ob_len = len(prompt_tokens) - 1  # same convention as rl_loop.py

                for tokens, logprobs, adv in zip(
                    group_tokens, group_logprobs, advantages
                ):
                    input_tokens = tokens[:-1]
                    target_tokens = tokens[1:]

                    # pad sampling logprobs / advantages over the prompt region with zeros
                    all_logprobs = [0.0] * ob_len + logprobs
                    all_advantages = [0.0] * ob_len + [adv] * (len(input_tokens) - ob_len)

                    assert (
                        len(input_tokens)
                        == len(target_tokens)
                        == len(all_logprobs)
                        == len(all_advantages)
                    ), "length mismatch when building Datum"

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens, dtype=torch.long)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(all_logprobs, dtype=torch.float32)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(all_advantages, dtype=torch.float32)
                            ),
                        },
                    )
                    training_datums.append(datum)

            if not training_datums:
                print(
                    f"[step {global_step}] No non-zero-advantage samples in batch; skipping update."
                )
                global_step += 1
                continue

            # ------------------------------------------------------------
            # 4) Tinker training step (importance_sampling)
            # ------------------------------------------------------------
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn="importance_sampling",
            )
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            _ = optim_future.result()

            # Extract total loss if you want it (optional)
            # loss_sums = [o["loss:sum"] for o in fwd_bwd_result.metrics]
            # ...

            # ------------------------------------------------------------
            # 5) Logging + checkpointing
            # ------------------------------------------------------------
            mean_batch_reward = (
                sum(batch_mean_rewards) / len(batch_mean_rewards)
                if batch_mean_rewards
                else 0.0
            )

            elapsed = time.time() - t_start
            print(
                f"[epoch {epoch} step {global_step}] "
                f"batch_size={len(batch_rows)}, "
                f"datums={len(training_datums)}, "
                f"mean_reward={mean_batch_reward:.4f}, "
                f"time={elapsed:.1f}s"
            )

            wandb.log(
                {
                    "reward/mean": mean_batch_reward,
                    "time/batch_sec": elapsed,
                    "optim/lr": LEARNING_RATE,
                    "train/datums": len(training_datums),
                    "progress/epoch": epoch,
                    "progress/step": global_step,
                },
                step=global_step,
            )

            # save Tinker state every SAVE_EVERY steps
            if SAVE_EVERY and (global_step + 1) % SAVE_EVERY == 0:
                state_name = f"{global_step + 1:06d}"
                print(f"Saving Tinker state: {state_name}")
                training_client.save_state(name=state_name).result()

            global_step += 1

    # final checkpoint
    print("Saving final Tinker state...")
    training_client.save_state(name="final").result()
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train_with_tinker()
