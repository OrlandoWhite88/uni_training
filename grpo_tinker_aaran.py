"""
Pure GRPO reinforcement learning on HS classification with Tinker.

Key corrections vs original code:
- Correct per-token advantage alignment
- Mask prompt region advantages/logprobs
- Remove KL, remove ref model, remove likelihood ratios
- Ensure Tinker importance_sampling receives correct tensors
- Stable reward climbing when reward ∈ [0, 1]
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

from reward import evaluate


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

MODEL_ID = "openai/gpt-oss-120b"
DATASET_PATH = Path("dataset.jsonl")

LORA_RANK = 32
BASE_URL: Optional[str] = None
RESUME_STATE_PATH: Optional[str] = None

BATCH_SIZE = 64
GROUP_SIZE = 4
MAX_PROMPT_TOKENS = 32000
MAX_COMPLETION_TOKENS = 2048
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1

TEMPERATURE = 1.1
TOP_P = 0.93

SAVE_EVERY = 10

SUPPORTED_TASKS = {"select_chapters", "select_candidates", "score_candidate"}


# ---------------------------------------------------------------------
# DATASET PREP  (same as yours)
# ---------------------------------------------------------------------

def prepare_grpo_dataset(example: Dict) -> Dict:
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
    try:
        user_data = json.loads(example["user_data"])
    except Exception:
        return False
    return user_data.get("task") in SUPPORTED_TASKS


def filter_by_length(example: Dict, tokenizer, max_tokens) -> bool:
    tokens = tokenizer.apply_chat_template(
        example["prompt"],
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens=False,
    )
    return len(tokens) <= max_tokens


def load_and_prepare_dataset(tokenizer) -> Dataset:
    dataset_path = DATASET_PATH.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    raw = datasets.load_dataset("json", data_files=str(dataset_path), split="train")

    ds = raw.map(
        prepare_grpo_dataset, remove_columns=raw.column_names
    )
    ds = ds.filter(filter_by_task_type)
    ds = ds.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "max_tokens": MAX_PROMPT_TOKENS},
    )
    if len(ds) == 0:
        raise ValueError("No examples remaining after filtering.")

    return ds


# ---------------------------------------------------------------------
# REWARD EXTRACTION
# ---------------------------------------------------------------------

def extract_last_json(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    last_open = text.rfind("{")
    if last_open == -1:
        return None
    brace_count = 0
    for i in range(last_open, len(text)):
        c = text[i]
        if c == "{":
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[last_open : i + 1]
    return None


def compute_reward(sampled_tokens: List[int], user_data, targets, tokenizer) -> float:
    try:
        decoded = tokenizer.decode(sampled_tokens)
        json_text = extract_last_json(decoded)
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


# ---------------------------------------------------------------------
# TINKER SETUP
# ---------------------------------------------------------------------

def create_training_client() -> tinker.TrainingClient:
    service_client = tinker.ServiceClient(base_url=BASE_URL)

    if RESUME_STATE_PATH:
        print(f"Resuming from {RESUME_STATE_PATH}")
        return service_client.create_training_client_from_state(RESUME_STATE_PATH)
    else:
        print(f"Starting new LoRA run from base {MODEL_ID}")
        return service_client.create_lora_training_client(
            base_model=MODEL_ID, rank=LORA_RANK
        )


# ---------------------------------------------------------------------
# GRPO TRAINING
# ---------------------------------------------------------------------

def train_with_tinker():
    wandb.init(project="hs-code-grpo", name="gpt-oss-20b-tinker-rl")

    training_client = create_training_client()
    tokenizer = training_client.get_tokenizer()

    dataset = load_and_prepare_dataset(tokenizer)
    n_examples = len(dataset)
    n_batches = math.ceil(n_examples / BATCH_SIZE)
    print(f"Dataset size = {n_examples}, batches/epoch = {n_batches}")

    sampling_params = types.SamplingParams(
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    adam_params = types.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        epoch_ds = dataset.shuffle(seed=epoch)
        print(f"=== Epoch {epoch+1}/{NUM_EPOCHS} ===")

        for batch_idx in range(n_batches):
            t0 = time.time()
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, n_examples)
            rows = epoch_ds.select(range(start, end))

            # ------------------------------------------------------------
            # 1) Freeze current model weighting for sampling
            # ------------------------------------------------------------
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"step_{global_step:06d}"
            )

            batch_prompt_tokens = []
            batch_user_data = []
            batch_targets = []
            batch_futures = []

            # ------------------------------------------------------------
            # 2) Prepare queries and sample GROUP_SIZE rollouts each
            # ------------------------------------------------------------
            for row in rows:
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

                futures = []
                for _ in range(GROUP_SIZE):
                    fut = sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    futures.append(fut)

                batch_prompt_tokens.append(prompt_tokens)
                batch_user_data.append(user_data)
                batch_targets.append(targets)
                batch_futures.append(futures)

            # ------------------------------------------------------------
            # 3) Compute rewards and build GRPO datums
            # ------------------------------------------------------------
            training_datums = []
            batch_mean_rewards = []

            for futures, prompt_tokens, user_data, targets in zip(
                batch_futures, batch_prompt_tokens, batch_user_data, batch_targets
            ):
                completions = []
                for fut in futures:
                    out = fut.result()
                    seq = out.sequences[0]
                    sampled_tokens = seq.tokens
                    sampled_logprobs = seq.logprobs  # list of floats
                    assert sampled_logprobs is not None

                    reward = compute_reward(
                        sampled_tokens, user_data, targets, tokenizer
                    )

                    completions.append(
                        {
                            "sampled_tokens": sampled_tokens,
                            "logprobs": sampled_logprobs,
                            "reward": reward,
                        }
                    )

                if len(completions) == 0:
                    continue

                rewards = [c["reward"] for c in completions]
                group_mean = sum(rewards) / len(rewards)
                batch_mean_rewards.append(group_mean)

                ob_len = len(prompt_tokens) - 1

                for c in completions:
                    reward = c["reward"]
                    logp = c["logprobs"]
                    adv = reward - group_mean

                    all_tokens = prompt_tokens + c["sampled_tokens"]

                    input_tokens = all_tokens[:-1]
                    target_tokens = all_tokens[1:]

                    full_len = len(input_tokens)
                    assert full_len == len(target_tokens)

                    # Mask prompt: zero logprobs, zero advantages
                    padded_logp = [0.0] * ob_len + logp
                    padded_adv = [0.0] * ob_len + [adv] * (full_len - ob_len)

                    assert len(padded_logp) == full_len
                    assert len(padded_adv) == full_len

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens, dtype=torch.long)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(padded_logp, dtype=torch.float32)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(padded_adv, dtype=torch.float32)
                            ),
                        },
                    )
                    training_datums.append(datum)

            if not training_datums:
                print(f"[step {global_step}] no non-zero advantages; skipping update.")
                global_step += 1
                continue

            # ------------------------------------------------------------
            # 4) GRPO update via importance_sampling loss
            # ------------------------------------------------------------
            fwd_bwd_fut = training_client.forward_backward(
                training_datums,
                loss_fn="importance_sampling",
            )
            opt_fut = training_client.optim_step(adam_params)

            _ = fwd_bwd_fut.result()
            _ = opt_fut.result()

            # ------------------------------------------------------------
            # 5) Logging + checkpointing
            # ------------------------------------------------------------
            mean_reward = (
                sum(batch_mean_rewards) / len(batch_mean_rewards)
                if batch_mean_rewards
                else 0.0
            )

            dt = time.time() - t0
            print(
                f"[epoch {epoch} step {global_step}] "
                f"datums={len(training_datums)}, "
                f"mean_reward={mean_reward:.4f}, "
                f"time={dt:.1f}s"
            )

            wandb.log(
                {
                    "reward/mean": mean_reward,
                    "train/datums": len(training_datums),
                    "progress/epoch": epoch,
                    "progress/step": global_step,
                    "time/batch": dt,
                },
                step=global_step,
            )

            if SAVE_EVERY and (global_step + 1) % SAVE_EVERY == 0:
                name = f"{global_step+1:06d}"
                print(f"Saving Tinker state {name}")
                training_client.save_state(name=name).result()

            global_step += 1

    print("Saving final state...")
    training_client.save_state(name="final").result()
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train_with_tinker()
