import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import wandb

import tinker
from tinker import types
from tinker.types import TensorData
from transformers import AutoTokenizer

from reward import evaluate


# ================================================================
# CONFIG
# ================================================================
BASE_MODEL = "openai/gpt-oss-120b"
DATASET_PATH = Path("dataset.jsonl")

MAX_PROMPT_TOKENS = 8000
MAX_COMPLETION_TOKENS = 2048

LEARNING_RATE = 5e-6
GRAD_ACCUM_STEPS = 4
NUM_GENERATIONS = 4
SAVE_STEPS = 10

SUPPORTED_TASKS = {"select_chapters", "select_candidates", "score_candidate"}


# ================================================================
# PROMPT RENDERER
# ================================================================
def render_messages(messages):
    sys = messages[0]["content"]
    usr = messages[1]["content"]
    return (
        "<system>\n" + sys + "\n</system>\n\n"
        "<user>\n" + usr + "\n</user>\n\n"
        "<assistant>\n"
    )


# ================================================================
# DATASET
# ================================================================
def prepare_example(ex):
    msgs = ex["messages"]
    return {
        "prompt": [
            {"role": "system", "content": msgs[0]["content"]},
            {"role": "user", "content": msgs[1]["content"]},
        ],
        "user_data": msgs[1]["content"],
        "targets": msgs[2]["content"],
    }


def filter_by_task(ex):
    try:
        ud = json.loads(ex["user_data"])
        return ud.get("task") in SUPPORTED_TASKS
    except:
        return False


def load_filtered_dataset(tokenizer):
    raw = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    ds = raw.map(prepare_example, remove_columns=raw.column_names)
    ds = ds.filter(filter_by_task)

    def ok(ex):
        toks = tokenizer.encode(render_messages(ex["prompt"]))
        return len(toks) <= MAX_PROMPT_TOKENS

    ds = ds.filter(ok)
    return ds


# ================================================================
# JSON EXTRACTION
# ================================================================
def extract_last_json(s):
    pos = s.rfind("{")
    if pos == -1:
        return None
    depth = 0
    for i in range(pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[pos:i+1]
    return None


def make_reward_fn():
    def fn(completions, _, user_datas, targets, **kwargs):
        rewards = []
        for comp, ud, tgt in zip(completions, user_datas, targets):
            js = extract_last_json(comp)
            if not js:
                rewards.append(0.0)
                continue
            ud = json.loads(ud)
            tgt = json.loads(tgt)
            res = evaluate(user_data=ud, answer=js, targets=tgt)
            score = res["score"] if res.get("is_score_valid") else 0.0
            rewards.append(float(score))
        return rewards
    return fn


# ================================================================
# MAIN TRAINING LOOP (PPO)
# ================================================================
def main():

    wandb.init(project="hs-code-grpo-tinker", name="grpo-lora-ppo")

    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training.get_tokenizer()

    dataset = load_filtered_dataset(tokenizer)
    print("Dataset size:", len(dataset))

    reward_fn = make_reward_fn()

    # Sampling clients:
    #  - actor_sampler = samples using UPDATED weights
    #  - baseline_sampler = samples using FROZEN weights (old policy)
    actor_sampler = training.save_weights_and_get_sampling_client("actor_init")
    baseline_sampler = training.save_weights_and_get_sampling_client("baseline_init")

    global_step = 0
    accum = 0

    for ex in dataset:

        msgs = ex["prompt"]
        ud = json.loads(ex["user_data"])
        tgt = json.loads(ex["targets"])

        prompt_txt = render_messages(msgs)
        prompt_ids = tokenizer.encode(prompt_txt)
        model_input = types.ModelInput.from_ints(prompt_ids)
        prompt_len = len(prompt_ids)

        params = types.SamplingParams(
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=1.1,
            top_p=0.9,
        )

        # =====================================================================
        # SAMPLE FROM ACTOR (CURRENT POLICY)
        # =====================================================================
        actor_out = actor_sampler.sample(model_input, NUM_GENERATIONS, params).result()
        actor_texts = [tokenizer.decode(seq.tokens) for seq in actor_out.sequences]

        # =====================================================================
        # SAMPLE FROM BASELINE (OLD POLICY)
        # =====================================================================
        base_out = baseline_sampler.sample(model_input, NUM_GENERATIONS, params).result()

        # =====================================================================
        # COMPUTE REWARDS
        # =====================================================================
        rewards = reward_fn(
            actor_texts,
            None,
            [json.dumps(ud)] * NUM_GENERATIONS,
            [json.dumps(tgt)] * NUM_GENERATIONS,
        )
        mean_r = float(np.mean(rewards))

        batch = []

        for i in range(NUM_GENERATIONS):

            # CONTINUATION LENGTHS
            actor_seq = actor_out.sequences[i]
            base_seq = base_out.sequences[i]

            actor_cont = len(actor_seq.tokens)
            base_cont = len(base_seq.tokens)

            # FULL EPISODE LENGTH (actor continuation defines training)
            full_len = prompt_len + actor_cont

            # PAD ARRAYS
            target_tokens = np.full(full_len, -100, dtype=np.int64)
            advantages    = np.zeros(full_len, dtype=np.float32)
            logprobs_np   = np.zeros(full_len, dtype=np.float32)
            old_logprobs  = np.zeros(full_len, dtype=np.float32)

            # Fill continuation ranges
            target_tokens[prompt_len:] = np.array(actor_seq.tokens, dtype=np.int64)
            logprobs_np[prompt_len:]   = np.array(actor_seq.logprobs, dtype=np.float32)

            # PPO advantage = reward - baseline – simplified GRPO style
            advantages[prompt_len:] = rewards[i] - mean_r

            # Baseline logprobs (only fill their continuation slice)
            old_logprobs[prompt_len : prompt_len + base_cont] = np.array(
                base_seq.logprobs, dtype=np.float32
            )

            datum = types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData(
                        data=target_tokens.tolist(),
                        dtype="int64",
                        shape=[full_len],
                    ),
                    "advantages": TensorData(
                        data=advantages.tolist(),
                        dtype="float32",
                        shape=[full_len],
                    ),
                    "logprobs": TensorData(
                        data=logprobs_np.tolist(),
                        dtype="float32",
                        shape=[full_len],
                    ),
                    "old_logprobs": TensorData(
                        data=old_logprobs.tolist(),
                        dtype="float32",
                        shape=[full_len],
                    ),
                },
            )

            batch.append(datum)

        # PPO update
        training.forward_backward(batch, loss_fn="ppo").result()

        accum += 1

        if accum >= GRAD_ACCUM_STEPS:

            training.optim_step(
                types.AdamParams(learning_rate=LEARNING_RATE)
            ).result()

            accum = 0
            global_step += 1

            wandb.log({"step": global_step, "mean_reward": mean_r})

            # Refresh actor sampler to use updated weights
            actor_sampler = training.save_weights_and_get_sampling_client(f"actor_{global_step}")

            if global_step % SAVE_STEPS == 0:
                state = training.save_state(f"step_{global_step}")
                print("Saved:", state.path)
                # keep baseline fixed until explicit swap
                # (alternatively: baseline_sampler = actor_sampler.clone…)

    final = training.save_state("final")
    print("Training complete:", final.path)
    wandb.finish()


if __name__ == "__main__":
    main()
