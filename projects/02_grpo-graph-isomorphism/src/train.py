import os
import csv
import argparse

BASE_KWARGS = {
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.1,
    "num_generations": 8,
    "temperature": 1.0,
    "mask_truncated_completions": True,
    "max_prompt_length": 256,
    "max_completion_length": 512,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "save_steps": 50,
    "logging_steps": 1,
    "log_completions": True,
    "bf16": True,
    # Data config
    "num_questions": 500,
    "difficulty": 3,  # Will be overridden by CLI or profiling
}

# Run 1: Anti-hacking
RUN1_CONFIG = {
    "run_name": "run1_antihack",
    "reward_type": "class_aware",
    "scale_rewards": "batch",
    "loss_type": "dapo",
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "beta": 0.04,
    "iso_ratio": 0.75,
    "max_steps": 300,
    **BASE_KWARGS,
}

# Run 2: Vanilla control
RUN2_CONFIG = {
    "run_name": "run2_vanilla",
    "reward_type": "composite",
    "scale_rewards": "group",
    "loss_type": "grpo",
    "beta": 0.0,
    "iso_ratio": 0.5,
    "max_steps": 300,
    **BASE_KWARGS,
}

# Run 2b: Isolation (Vanilla + batch norm)
RUN2B_CONFIG = {
    "run_name": "run2b_batch_norm_only",
    "reward_type": "composite",
    "scale_rewards": "batch",  # ONLY CHANGE
    "loss_type": "grpo",
    "beta": 0.0,
    "iso_ratio": 0.5,
    "max_steps": 300,
    **BASE_KWARGS,
}

# Smoke test: Very short Run 1 copy
SMOKE_CONFIG = {
    **RUN1_CONFIG,
    "run_name": "smoke_test",
    "max_steps": 3,
    "save_steps": 3,
}

# Run H1: Hints experiment — reveal k=n-2 mappings so model only guesses 2
# Goal: bootstrap nonzero iso success rate → GRPO has signal
RUN_H1_CONFIG = {
    **BASE_KWARGS,
    "run_name": "run_h1_hints",
    "reward_type": "class_aware",
    "scale_rewards": "batch",
    "loss_type": "grpo",
    "beta": 0.0,
    "iso_ratio": 0.75,
    "max_steps": 200,
    "difficulty": 3,       # 4 nodes
    "hints_k": "n-2",     # reveal n-2 mappings (2 of 4 at diff 3)
}

# Run S1: SFT warmup (Phase A) → GRPO (Phase B)
# Phase A: sft_warmup.py saves adapter to sft_adapter_path
# Phase B: this config loads that adapter then runs GRPO
RUN_S1_CONFIG = {
    **BASE_KWARGS,
    "run_name": "run_s1_sft_grpo",
    "reward_type": "class_aware",
    "scale_rewards": "batch",
    "loss_type": "grpo",
    "beta": 0.0,
    "iso_ratio": 0.75,
    "max_steps": 200,
    "difficulty": 2,       # 4 nodes, easier
    "sft_adapter_path": "outputs/run_s1_sft",  # from sft_warmup.py Phase A
}

# Edge Counting positive control — proves GRPO pipeline works
# v1 (200 steps, lr=5e-6, 8 gens): no learning. v2 (lr=2e-5): diverged.
# v3: original lr + tight grad clip + more gens/steps.
RUN_EC_CONFIG = {
    **BASE_KWARGS,
    "run_name": "run_ec",
    "task": "edge_counting",
    "reward_type": "ec",
    "scale_rewards": "batch",
    "loss_type": "grpo",
    "beta": 0.0,
    "max_steps": 500,
    "difficulty": 1,                     # 3-node graphs — easiest
    "num_generations": 16,               # 2x more for better variance
    "learning_rate": 5e-6,               # back to original (2e-5 diverged)
    "max_grad_norm": 0.01,               # tighter clipping as safety
    "max_completion_length": 128,        # EC answers are very short
    "max_prompt_length": 256,            # prompts are ~160 tokens
    "mask_truncated_completions": False,  # workaround unsloth shape bug
}

# Brute-Force positive control — 3-node GI with 32 generations
# Hypothesis: with 32 samples, random chance produces valid mapping
# for 3-node graphs (only 6 permutations) → nonzero variance → GRPO signal.
# VRAM budget (24GB): Qwen2.5-1.5B 4-bit ≈ 3GB, 32 × 200 tok KV ≈ 0.5GB → fits.
RUN_BF_CONFIG = {
    **{k: v for k, v in BASE_KWARGS.items() if k not in [
        "num_generations", "max_completion_length",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "mask_truncated_completions",
    ]},
    "run_name": "run_bf",
    "reward_type": "class_aware",
    "scale_rewards": "batch",
    "loss_type": "grpo",
    "beta": 0.0,
    "iso_ratio": 0.75,
    "max_steps": 150,
    "difficulty": 1,               # 3 nodes — only 6 permutations
    "num_generations": 32,         # brute-force exploration
    "max_completion_length": 200,  # short outputs expected
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mask_truncated_completions": False,  # workaround unsloth shape bug
}

CONFIGS = {
    "run1": RUN1_CONFIG,
    "run2": RUN2_CONFIG,
    "run2b": RUN2B_CONFIG,
    "smoke": SMOKE_CONFIG,
    "h1": RUN_H1_CONFIG,
    "s1": RUN_S1_CONFIG,
    "ec": RUN_EC_CONFIG,
    "bf": RUN_BF_CONFIG,
}

from data_utils import prepare_dataset
from edge_counting_env import prepare_ec_dataset


def run_training(config):
    # Guard GPU imports
    import torch
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from rewards import class_aware_reward, composite_reward
    from edge_counting_reward import ec_reward

    # os.environ['UNSLOTH_VLLM_STANDBY'] = '1'
    # trl doesn't pass completion text to callbacks — patch it through
    class _LogCompletionsTrainer(GRPOTrainer):
        def log(self, logs, start_time=None):
            if hasattr(self, "_logs") and self._logs.get("completion"):
                logs["completions"] = list(self._logs["completion"])
            super().log(logs, start_time)

        def compute_loss(self, model, inputs, *args, **kwargs):
            # Workaround: unsloth compiled GRPOTrainer has a shape mismatch
            # between completion_mask (from forward pass) and coef_1/logprob
            # tensors (from vLLM generation). Align all seq-dim tensors to
            # the completion_mask length before calling super().
            cm = inputs.get("completion_mask")
            if cm is not None:
                seq_len = cm.shape[1]
                for key in ("old_per_token_logps", "ref_per_token_logps",
                            "advantages"):
                    t = inputs.get(key)
                    if t is not None and t.dim() >= 2 and t.shape[1] != seq_len:
                        inputs[key] = t[:, :seq_len]
                # Also align completion_ids so everything is consistent
                cids = inputs.get("completion_ids")
                if cids is not None and cids.shape[1] != seq_len:
                    inputs["completion_ids"] = cids[:, :seq_len]
            return super().compute_loss(model, inputs, *args, **kwargs)

    os.environ['WANDB_MODE'] = 'disabled'

    run_name = config["run_name"]
    output_dir = os.path.join("outputs", run_name)

    task = config.get("task", "graph_isomorphism")
    hints_k = config.get("hints_k")
    sft_adapter_path = config.get("sft_adapter_path")
    print(f"=== {run_name} | task={task} diff={config.get('difficulty')} n={config.get('num_questions')} ===")

    if task == "edge_counting":
        train_dataset = prepare_ec_dataset(config=config)
    else:
        train_dataset = prepare_dataset(config=config)

    if config["reward_type"] == "ec":
        reward_funcs = [ec_reward]
    elif config["reward_type"] == "class_aware":
        reward_funcs = [class_aware_reward]
    else:
        reward_funcs = [composite_reward]

    if sft_adapter_path and os.path.isdir(sft_adapter_path):
        # S1 Phase B: load the SFT-warmed adapter instead of base model
        print(f"Loading SFT adapter from {sft_adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter_path,
            max_seq_length=1024,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=32,
            gpu_memory_utilization=0.6,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
            max_seq_length=1024,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=32,
            gpu_memory_utilization=0.6,
        )
    
    model = FastLanguageModel.get_peft_model(
        model, 
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # strip custom keys not in GRPOConfig
    grpo_kwargs = {k: v for k, v in config.items() if k not in [
        "run_name", "reward_type", "iso_ratio", "num_questions", "difficulty",
        "hints_k", "sft_adapter_path", "task",
    ]}
    grpo_kwargs["output_dir"] = output_dir
    grpo_kwargs["log_on_each_node"] = False
    
    if grpo_kwargs.get("loss_type") == "grpo":
        grpo_kwargs.pop("epsilon_high", None)

    training_args = GRPOConfig(**grpo_kwargs)

    from callbacks import CollapseDetector
    detector = CollapseDetector(run_name)

    trainer = _LogCompletionsTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[detector],
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.state.log_history
    metrics_path = os.path.join(output_dir, "train_metrics.csv")
    if metrics:
        # union all keys — later entries (e.g. train_runtime) have fields the first doesn't
        all_keys = list(dict.fromkeys(
            k for m in metrics if isinstance(m, dict) for k in m
        ))
        if all_keys:
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for m in metrics:
                    if isinstance(m, dict):
                        writer.writerow(m)

    print(f"=== Done {run_name} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--iso_ratio", type=float)
    parser.add_argument("--hints_k", type=str)

    args = parser.parse_args()

    config = CONFIGS[args.config].copy()
    if args.difficulty is not None:
        config["difficulty"] = args.difficulty
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.iso_ratio is not None:
        config["iso_ratio"] = args.iso_ratio
    if args.hints_k is not None:
        config["hints_k"] = args.hints_k

    run_training(config)

if __name__ == "__main__":
    main()
