#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO trainer for GPT-OSS-120B (Unsloth) with:
- Flex attention for training (fast)
- Automatic eager fallback during generation (no OOM)
- Raw message-list parsing preserved (no change to your dataset format)

Launch (single GPU):
  python3 train.py

Launch (multi-GPU with accelerate):
  accelerate launch train.py
"""

# -------------------- ENV --------------------
import os

# Prefer Flex Attention everywhere; let eval fallback to eager
os.environ.setdefault("UNSLOTH_FLEX_ATTENTION", "1")
# Disable FlashAttention so Flex/eager are the only paths
os.environ.setdefault("UNSLOTH_DISABLE_FLASH_ATTENTION", "1")
# Help against fragmentation at long context
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------- Imports ----------------
# IMPORTANT: unsloth BEFORE transformers/trl
from unsloth import FastLanguageModel

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from transformers import GenerationConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

try:
    from accelerate import PartialState
    _ACCEL = True
except Exception:
    PartialState = None
    _ACCEL = False


# -------------------- Knobs ------------------
MODEL_NAME = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
DATASET_PATH = "dataset.jsonl"
REWARD_PATH = "reward.py"

# Adapters / resume
START_ADAPTER = "uni_grpo_model"           # change if needed
WORKING_ADAPTER = "sft-adapter-12-grpo"
RESUME_FROM_CHECKPOINT = "sft-adapter-12-grpo/checkpoint-120"  # or None

# Context / tokens (tight like the good run)
CTX_WINDOW = 5514
PROMPT_TOKEN_CAP = 4864  # leave budget for completion
COMPLETION_TOKEN_CAP = 550

# GRPO trainer batch/gen
NUM_GENERATIONS = 3
PER_DEVICE_TRAIN_BS = 3         # keep divisible by NUM_GENERATIONS
GRAD_ACC = 8
NUM_EPOCHS = 1
MAX_STEPS = -1                      # epochs-driven

LEARNING_RATE = 5e-7
BF16 = True
REASONING_EFFORT = "high"         # match your working script

# Logging / saving
DEBUG_COMPLETIONS_FILE = "debug_grpo_completions.jsonl"
DEBUG_FIRST_N = 50
LOGGING_STEPS = 1
SAVE_STEPS = 5
WARMUP_STEPS = 250

SUPPORTED_TASKS = {"score_candidate", "select_candidates", "select_chapters"}


# -------------------- Utils ------------------
def _install_bf16_eager_softmax():
    """
    Keep eager softmax in BF16 to avoid FP32 temp spikes during generation.
    """
    try:
        import unsloth_zoo.temporary_patches.gpt_oss as _unsloth_gpt_oss
        import transformers.models.gpt_oss.modeling_gpt_oss as _hf_gpt_oss
    except Exception:
        return

    if getattr(_unsloth_gpt_oss, "_bf16_softmax_installed", False):
        return

    base_softmax = torch.nn.functional.softmax

    def _softmax_keep_dtype(input, dim=None, _stacklevel=3, dtype=None):
        target_dtype = input.dtype if dtype in (None, torch.float32) else dtype
        return base_softmax(input, dim=dim, dtype=target_dtype)

    _unsloth_gpt_oss.F_softmax = _softmax_keep_dtype
    if hasattr(_hf_gpt_oss, "F_softmax"):
        _hf_gpt_oss.F_softmax = _softmax_keep_dtype
    _unsloth_gpt_oss._bf16_softmax_installed = True


def _flex_mask_pre_hook(module, args, kwargs):
    """
    TRAINING (module.training=True): coerce attention_mask -> 2D bool to keep Flex kernels fast.
    EVAL / GENERATION (module.training=False): DO NOT TOUCH MASKS.
        Leaving 4D additive masks intact allows Unsloth to auto-fallback to eager safely.
    Also always drop output_attentions to avoid slow branches.
    """
    kwargs.pop("output_attentions", None)

    if not module.training:
        # generation path -> eager fallback is decided by mask shape; don't modify
        return args, kwargs

    am = kwargs.get("attention_mask", None)
    if isinstance(am, torch.Tensor):
        if am.dim() != 2:
            am = am.view(am.size(0), -1)
        if am.dtype != torch.bool:
            am = (am > 0)
        kwargs["attention_mask"] = am

    return args, kwargs


# -------------------- Dataset (raw message lists) ------------------
_HARMONY_CHANNEL_PATTERN = re.compile(
    r"<\|channel\|>(?P<header>[^<]+)<\|message\|>(?P<body>.*?)(?=(?:<\|channel\|>|<\|end\|>|<\|return\|>|$))",
    re.DOTALL,
)

def load_harmony_jsonl(path: str) -> Dataset:
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset

def _strip_thinking(message: Dict[str, Any]) -> Dict[str, str]:
    role = message.get("role")
    content = message.get("content", "")
    if isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=False)
    return {"role": role, "content": str(content)}

def prepare_grpo_dataset(examples: dict, tokenizer=None, reasoning_effort: str = "medium") -> dict:
    """
    Preserve RAW MESSAGE LISTS as 'prompt' (what you asked for).
    """
    messages_batch: List[List[dict]] = examples["messages"]

    prompts: List[List[Dict[str, str]]] = []
    target_answers: List[str] = []
    user_data_list: List[str] = []
    targets_list: List[str] = []
    tasks: List[str] = []

    for messages in messages_batch:
        prompt_messages: List[Dict[str, str]] = []
        assistant_message: Optional[Dict[str, Any]] = None

        for message in messages:
            role = message.get("role", "")
            if role in ["system", "user"]:
                prompt_messages.append(message)
            elif role == "assistant" and assistant_message is None:
                assistant_message = message

        prompts.append(prompt_messages)

        assistant_content = ""
        if assistant_message is not None:
            raw_content = assistant_message.get("content", "")
            if isinstance(raw_content, dict):
                assistant_content = json.dumps(raw_content, ensure_ascii=False)
            else:
                assistant_content = str(raw_content)
        target_answers.append(assistant_content)

        try:
            assistant_data = json.loads(assistant_content) if assistant_content else {}
        except Exception:
            assistant_data = {"raw_content": assistant_content}

        user_payload: Optional[str] = None
        for msg in prompt_messages:
            if msg.get("role") == "user":
                user_payload = msg.get("content", "")
                break

        if user_payload:
            try:
                parsed_user = json.loads(user_payload)
                user_data_list.append(json.dumps(parsed_user, ensure_ascii=False))
                tasks.append(parsed_user.get("task", "unknown"))
            except Exception:
                user_data_list.append(user_payload)
                tasks.append("unknown")
        else:
            user_data_list.append("{}")
            tasks.append("unknown")

        targets_list.append(json.dumps(assistant_data, ensure_ascii=False))

    return {
        "prompt": prompts,                  # <— raw message lists kept
        "target_answer": target_answers,
        "user_data": user_data_list,
        "targets": targets_list,
        "task": tasks,
    }

def is_within_token_budget(example: dict, tokenizer, max_prompt_length: int, reasoning_effort: str = "medium") -> bool:
    """Use chat template over raw message list to count tokens."""
    try:
        prompt_messages = example.get("prompt", [])
        if not prompt_messages:
            return False

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        token_count = len(tokenizer.encode(prompt_text, add_special_tokens=True))
        return token_count <= max_prompt_length
    except Exception:
        return False

def build_grpo_dataset(tokenizer, dataset_path: str, reasoning_effort: str, is_main: bool) -> Dataset:
    if is_main:
        print(f"[dataset] Loading from {dataset_path}...")

    raw_dataset = load_harmony_jsonl(dataset_path)

    processed_dataset = raw_dataset.map(
        prepare_grpo_dataset,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Preparing GRPO dataset (raw messages)",
    )

    initial_size = len(processed_dataset)
    
    # Filter out unrecognized task types to prevent zero scores
    processed_dataset = processed_dataset.filter(
        lambda example: example.get("task") in SUPPORTED_TASKS,
        desc="Filtering unsupported task types",
    )
    
    if is_main:
        task_filtered = initial_size - len(processed_dataset)
        if task_filtered > 0:
            print(f"[dataset] Filtered {task_filtered:,} examples with unsupported task types")
    
    # Filter by token budget
    budget_initial = len(processed_dataset)
    processed_dataset = processed_dataset.filter(
        lambda example: is_within_token_budget(
            example,
            tokenizer,
            PROMPT_TOKEN_CAP,
            reasoning_effort=reasoning_effort,
        ),
        desc="Filtering by token budget",
    )

    if is_main:
        budget_filtered = budget_initial - len(processed_dataset)
        print(f"[dataset] Filtered {budget_filtered:,} examples exceeding token budget")
        print(f"[dataset] Kept {len(processed_dataset):,} examples within budget")

    return processed_dataset


# -------------------- Reward wrapper ------------------
def load_reward_fn(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("grpo_reward_module", path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load reward module from {path}")
    spec.loader.exec_module(mod)
    if not hasattr(mod, "evaluate"):
        raise AttributeError("reward.py must expose evaluate(user_data, answer, targets)->dict.")
    return getattr(mod, "evaluate")

def make_reward_wrapper(evaluate_fn, num_generations: int, debug_file: str, debug_first_n: int):
    import json
    MAIN_RANK = os.getenv("LOCAL_RANK", "0") in ("0", "", None)

    make_reward_wrapper._debug_count = getattr(make_reward_wrapper, "_debug_count", 0)

    def _is_seq(x: Any) -> bool:
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _extract_json_snippet(text: str, task_hint: Optional[str]) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None

        if text.startswith("{") and text.endswith("}"):
            try:
                json.loads(text)
                return text
            except Exception:
                pass

        final_marker = "<|channel|>final<|message|>"
        if final_marker in text:
            tail = text.split(final_marker)[-1]
            decoder = json.JSONDecoder()
            for m in json.decoder.re.finditer(r"\{", tail):
                try:
                    obj, end = decoder.raw_decode(tail[m.start():])
                    if isinstance(obj, dict):
                        return tail[m.start(): m.start() + end]
                except Exception:
                    continue

        decoder = json.JSONDecoder()
        found = []
        for m in json.decoder.re.finditer(r"\{", text):
            try:
                obj, end = decoder.raw_decode(text[m.start():])
                if isinstance(obj, dict):
                    found.append((obj, text[m.start(): m.start() + end]))
            except Exception:
                continue
        if not found:
            return None

        expected = {
            "select_chapters": ["chapters"],
            "select_candidates": ["selected_indices"],
            "score_candidate": ["option_number", "confidence"],
        }
        if task_hint in expected:
            for obj, snippet in reversed(found):
                if any(f in obj for f in expected[task_hint]):
                    return snippet
        for obj, snippet in reversed(found):
            if any(f in obj for fields in expected.values() for f in fields):
                return snippet
        return found[-1][1]

    def reward_fn(completions: List[Any], **kwargs) -> List[float]:
        user_entries = kwargs.get("user_data") or []
        target_entries = kwargs.get("targets") or []
        prompts = kwargs.get("prompts") or []
        grouped = bool(completions) and _is_seq(completions[0])

        if MAIN_RANK and not hasattr(reward_fn, "_once"):
            print(f"[reward] First call - kwargs keys: {list(kwargs.keys())}")
            print(f"[reward] Completions grouped: {grouped}")
            print(f"[reward] Num user_data: {len(user_entries)}")
            print(f"[reward] Num targets: {len(target_entries)}")
            reward_fn._once = True

        if grouped:
            batch_count = len(completions)
        else:
            total = len(completions)
            batch_count = max(1, total // max(1, num_generations))

        rewards: List[float] = []
        for b in range(batch_count):
            if grouped:
                gen_group = completions[b]
            else:
                s = b * num_generations
                e = s + num_generations
                gen_group = completions[s:e]

            user_str = user_entries[b] if b < len(user_entries) else "{}"
            target_str = target_entries[b] if b < len(target_entries) else "{}"
            prompt_str = prompts[b] if b < len(prompts) else ""

            try:
                user_data = json.loads(user_str) if isinstance(user_str, str) else user_str
            except Exception:
                user_data = {}
            try:
                targets = json.loads(target_str) if isinstance(target_str, str) else target_str
            except Exception:
                targets = {}

            def _stringify(gen):
                if isinstance(gen, dict):
                    return str(gen.get("content", "")).strip()
                if _is_seq(gen):
                    return "\n".join(str(x.get("content", "") if isinstance(x, dict) else x) for x in gen).strip()
                return str(gen).strip()

            for g_idx, completion in enumerate(gen_group):
                answer_text = _stringify(completion)
                json_snip = _extract_json_snippet(answer_text, user_data.get("task"))
                answer_for_reward = json_snip if json_snip else answer_text

                try:
                    result = evaluate_fn(user_data, answer_for_reward, targets)
                    score = float(result.get("score", 0.0))
                    reason = str(result.get("reason", ""))
                    is_valid = bool(result.get("is_score_valid", False))
                except Exception as exc:
                    score = 0.0
                    reason = f"Exception in reward.py: {exc}"
                    is_valid = False

                rewards.append(max(0.0, min(1.0, score)))

                if MAIN_RANK and make_reward_wrapper._debug_count < debug_first_n:
                    make_reward_wrapper._debug_count += 1
                    idx = make_reward_wrapper._debug_count
                    print("\n" + "=" * 80)
                    print(f"[GRPO DEBUG #{idx}] Batch {b + 1}, Generation {g_idx + 1}/{len(gen_group)} | Task: {user_data.get('task', 'unknown')}")
                    print("=" * 80)
                    print(f"Prompt (tail): {str(prompt_str)[-200:] or None}")
                    print("\n--- USER DATA ---")
                    print(json.dumps(user_data, indent=2)[:500])
                    print("\n--- TARGETS ---")
                    print(json.dumps(targets, indent=2)[:500])
                    print(f"\n--- MODEL OUTPUT #{g_idx + 1} (raw, {len(answer_text)} chars) ---")
                    print(answer_text[:1200] + (f"... [truncated, total {len(answer_text)} chars]" if len(answer_text) > 1200 else ""))
                    print(f"\n--- EXTRACTED JSON (success={json_snip is not None}) ---")
                    print((json_snip or "(none)")[:600])
                    print("\n--- REWARD ---")
                    print(f"Score: {score:.4f}")
                    print(f"Reason: {reason[:500]}")
                    print(f"Valid: {is_valid}")
                    print("=" * 80 + "\n")

                    try:
                        with open(DEBUG_COMPLETIONS_FILE, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({
                                "idx": idx,
                                "batch": b,
                                "gen": g_idx,
                                "task": user_data.get("task"),
                                "prompt_tail": str(prompt_str)[-200:] if prompt_str else None,
                                "user_data": user_data,
                                "targets": targets,
                                "raw_completion": answer_text[:1500],
                                "extracted_json": json_snip,
                                "json_extracted_success": json_snip is not None,
                                "completion_length": len(answer_text),
                                "reward_score": score,
                                "reward_reason": reason,
                                "is_valid": is_valid,
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
        return rewards
    return reward_fn


class MemoryManagementCallback(TrainerCallback):
    """Clear CUDA cache regularly to avoid memory creep."""
    def __init__(self):
        self.peak_memory = 0
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and state.global_step > 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            if mem_allocated > self.peak_memory:
                self.peak_memory = mem_allocated
            print(f"[memory] Step {state.global_step}: "
                  f"allocated={mem_allocated:.1f}GB, "
                  f"reserved={mem_reserved:.1f}GB, "
                  f"peak={self.peak_memory:.1f}GB")
        return control


class WandBMetricsCallback(TrainerCallback):
    """Log custom metrics to WandB during training."""
    def __init__(self):
        self.is_main = os.getenv("LOCAL_RANK", "0") in ("0", "", None)
        self.last_logged_step = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main or logs is None:
            return
        
        # Only log once per step to avoid wandb step conflicts
        if state.global_step == self.last_logged_step:
            return
        
        self.last_logged_step = state.global_step
        
        # Log custom metrics to wandb without specifying step
        # (let wandb auto-increment to avoid conflicts)
        wandb_logs = {}
        
        # Extract reward metrics if available
        if "rewards/mean" in logs:
            wandb_logs["custom/reward_mean"] = logs["rewards/mean"]
        if "rewards/std" in logs:
            wandb_logs["custom/reward_std"] = logs["rewards/std"]
        
        # Log any GRPO-specific metrics
        for key, value in logs.items():
            if key.startswith("rewards/") or key.startswith("grpo/"):
                wandb_logs[f"custom/{key}"] = value
        
        # Log to wandb - let wandb handle step counting automatically
        if wandb_logs and wandb.run is not None:
            wandb.log(wandb_logs)
        
        return control


# -------------------- Main -------------------
def main():
    assert torch.cuda.is_available(), "CUDA not available."

    if _ACCEL:
        state = PartialState()
        MAIN = state.is_main_process
    else:
        MAIN = os.getenv("LOCAL_RANK", "0") in ("0", "", None)

    def log(*args, **kwargs):
        if MAIN:
            print(*args, **kwargs)

    # Initialize wandb on main process only
    if MAIN:
        wandb.init(
            project="grpo-gpt-oss-120b",
            name=f"{WORKING_ADAPTER}-{REASONING_EFFORT}",
            config={
                "model": MODEL_NAME,
                "dataset": DATASET_PATH,
                "start_adapter": START_ADAPTER,
                "working_adapter": WORKING_ADAPTER,
                "resume_from": RESUME_FROM_CHECKPOINT,
                "ctx_window": CTX_WINDOW,
                "prompt_token_cap": PROMPT_TOKEN_CAP,
                "completion_token_cap": COMPLETION_TOKEN_CAP,
                "num_generations": NUM_GENERATIONS,
                "per_device_train_bs": PER_DEVICE_TRAIN_BS,
                "grad_accumulation": GRAD_ACC,
                "num_epochs": NUM_EPOCHS,
                "max_steps": MAX_STEPS,
                "learning_rate": LEARNING_RATE,
                "bf16": BF16,
                "reasoning_effort": REASONING_EFFORT,
                "logging_steps": LOGGING_STEPS,
                "save_steps": SAVE_STEPS,
                "warmup_steps": WARMUP_STEPS,
            },
            resume="allow",
        )
        log("[wandb] Initialized with project: grpo-gpt-oss-120b")

    log(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
    has_flex = hasattr(torch.nn.attention, "flex_attention")
    log("[check] flex_attention present:", has_flex)

    per_device_train_bs = PER_DEVICE_TRAIN_BS
    if per_device_train_bs % NUM_GENERATIONS != 0:
        log(f"[warn] Adjusting batch size from {per_device_train_bs} to {NUM_GENERATIONS}")
        per_device_train_bs = NUM_GENERATIONS

    log(f"[model] Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=torch.bfloat16,
        max_seq_length=CTX_WINDOW,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Flex globally; pre-hook enforces 2D masks only for training
    model.config.attn_implementation = "flex_attention"
    model.register_forward_pre_hook(_flex_mask_pre_hook, with_kwargs=True)
    _install_bf16_eager_softmax()  # make eager softmax BF16-safe

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = CTX_WINDOW
    log(f"[tokenizer] Vocab: {len(tokenizer)}, Pad: {tokenizer.pad_token}, EOS: {tokenizer.eos_token}")

    # Monkey-patch to default reasoning_effort for chat template calls
    original_apply_chat_template = tokenizer.apply_chat_template
    def patched_apply_chat_template(*args, **kwargs):
        if 'reasoning_effort' not in kwargs:
            kwargs['reasoning_effort'] = REASONING_EFFORT
        return original_apply_chat_template(*args, **kwargs)
    tokenizer.apply_chat_template = patched_apply_chat_template

    # Load LoRA adapter (resume or start)
    from peft import PeftModel
    if RESUME_FROM_CHECKPOINT and Path(RESUME_FROM_CHECKPOINT).exists():
        adapter_path = RESUME_FROM_CHECKPOINT
        log(f"[adapter] Resuming from checkpoint: {adapter_path}")
    elif Path(START_ADAPTER).exists():
        adapter_path = START_ADAPTER
        log(f"[adapter] Loading from: {adapter_path}")
    else:
        raise FileNotFoundError(f"Adapter not found: {START_ADAPTER}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    # Optional: sparse MoE forward (matches your working approach)
    import types
    from unsloth_zoo.temporary_patches import gpt_oss as _unsloth_gpt_oss
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    def _sparse_forward(self, hidden_states, router_indices=None, routing_weights=None):
        if router_indices is None or routing_weights is None:
            raise ValueError("router_indices and routing_weights must be provided.")
        bsz, seqlen, hidden = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden)
        flat_weights = routing_weights.reshape(-1, routing_weights.shape[-1])
        accum = flat_states.new_zeros(flat_states.size(0), hidden, dtype=torch.float32)
        active_experts = torch.unique(router_indices, sorted=False)
        for expert_idx in active_experts.tolist():
            mask = (router_indices == expert_idx)
            if not mask.any():
                continue
            flat_idx = mask.nonzero(as_tuple=False)[:, 0]
            tokens = flat_states.index_select(0, flat_idx)
            gate_up = self.gate_up_projs[expert_idx](tokens)
            fused = _unsloth_gpt_oss.swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype=tokens.dtype)
            expert_out = self.down_projs[expert_idx](fused)
            weights = flat_weights.index_select(0, flat_idx)[:, expert_idx].unsqueeze(-1).to(expert_out.dtype)
            accum.index_add_(0, flat_idx, (expert_out * weights).to(torch.float32))
        return accum.view(bsz, seqlen, hidden).to(hidden_states.dtype)

    patched = 0
    for module in model.modules():
        if isinstance(module, GptOssExperts):
            module.forward = types.MethodType(_sparse_forward, module)
            patched += 1
    log(f"[patch] Sparse MoE applied to {patched} expert blocks")

    try:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e:
        log(f"[warn] Could not enable gradient checkpointing: {e}")

    Path(WORKING_ADAPTER).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(WORKING_ADAPTER)
    log(f"[adapter] Copied to working dir: {WORKING_ADAPTER}")

    # Generation config — keep flex; eval mask forces eager fallback
    gen = GenerationConfig.from_model_config(model.config)
    gen.max_length = None
    gen.max_new_tokens = COMPLETION_TOKEN_CAP
    gen.use_cache = True
    gen.attn_implementation = "flex_attention"
    gen.temperature = 1.0
    gen.top_p = 0.92
    gen.top_k = 40
    gen.do_sample = True
    model.generation_config = gen
    model.config.use_cache = True

    # Dataset
    ds = build_grpo_dataset(tokenizer, DATASET_PATH, REASONING_EFFORT, MAIN)
    if MAIN and len(ds) > 0:
        log(f"\n[dataset] Example 1:")
        log(f"  Task: {ds[0]['task']}")
        log(f"  Prompt[0] role: {ds[0]['prompt'][0]['role'] if ds[0]['prompt'] else 'N/A'}")
        log(f"  User data (head): {ds[0]['user_data'][:200]}...")
        log(f"  Targets (head): {ds[0]['targets'][:200]}...")

    # Reward
    log(f"\n[reward] Loading from {REWARD_PATH}...")
    evaluate_fn = load_reward_fn(REWARD_PATH)
    reward_fn = make_reward_wrapper(evaluate_fn, NUM_GENERATIONS, DEBUG_COMPLETIONS_FILE, DEBUG_FIRST_N)
    log(f"[reward] Will log first {DEBUG_FIRST_N} completions to {DEBUG_COMPLETIONS_FILE}")

    # GRPO args (generation kwargs keep 'flex'; mask causes eager)
    grpo_args = GRPOConfig(
        output_dir=WORKING_ADAPTER,
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=GRAD_ACC,
        num_generations=NUM_GENERATIONS,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        warmup_steps=WARMUP_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to=["tensorboard", "wandb"],
        logging_dir=f"{WORKING_ADAPTER}/logs",
        max_prompt_length=PROMPT_TOKEN_CAP,
        max_completion_length=COMPLETION_TOKEN_CAP,
        mask_truncated_completions=True,
        scale_rewards="batch",
        generation_kwargs={
            "temperature": 1.0,
            "top_p": 0.92,
            "top_k": 40,
            "do_sample": True,
            "attn_implementation": "flex_attention",
            "output_attentions": False,
            "use_cache": True,
        },
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # Add callbacks
    trainer.add_callback(MemoryManagementCallback())
    if MAIN:
        trainer.add_callback(WandBMetricsCallback())
        log("[wandb] Added custom metrics callback")

    # Sanity logs
    log(f"[config] model.config.attn_implementation={model.config.attn_implementation}")
    log(f"[config] model.generation_config.attn_implementation={model.generation_config.attn_implementation}")
    log("[note] In eval, pre-hook leaves attention_mask intact -> Unsloth flex auto-fallbacks to eager.")

    log("\n" + "=" * 80)
    log("STARTING GRPO TRAINING")
    log("=" * 80)
    log(f"Dataset size: {len(ds):,}")
    log(f"Batch size: {per_device_train_bs}")
    log(f"Generations per prompt: {NUM_GENERATIONS}")
    log(f"Max steps: {MAX_STEPS}")
    log(f"Learning rate: {LEARNING_RATE}")
    log(f"Warmup steps: {WARMUP_STEPS}")
    if SAVE_STEPS != 5:
        raise ValueError("SAVE_STEPS must remain 5 to satisfy checkpoint cadence requirements.")
    log(f"Checkpoint save cadence: every {SAVE_STEPS} steps (strategy=steps)")

    # Resume if requested
    resume_path = RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT and Path(RESUME_FROM_CHECKPOINT).exists() else None
    if resume_path:
        log(f"Resuming from: {resume_path}")

    train_result = trainer.train(resume_from_checkpoint=resume_path)

    if MAIN:
        log("\n" + "=" * 80)
        log("TRAINING COMPLETE")
        log("=" * 80)
        log(train_result)

        final_dir = "grpo-gpt-oss-120b-final"
        Path(final_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        log(f"\n[save] Final adapter saved to: {final_dir}")
        
        # Finish wandb run
        wandb.finish()
        log("[wandb] Run finished and logged")


if __name__ == "__main__":
    main()
