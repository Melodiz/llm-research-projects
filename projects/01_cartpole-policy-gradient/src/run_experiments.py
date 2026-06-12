"""HW1 experiment orchestrator: PG variants + BC on CartPole-v1."""

import argparse
import copy
import json
import os
import sys
import traceback
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cartpole_pg import (
    Config,
    PolicyNetwork,
    train,
    evaluate,
    bc_failure_experiments,
    load_results,
    loss_vanilla_pg,
    loss_pg_avg_baseline,
    loss_pg_value_baseline,
    loss_rloo,
)


METHODS_CORE = {
    "vpg":               (loss_vanilla_pg,        False, False),
    "vpg_avg":           (loss_pg_avg_baseline,   False, False),
    "vpg_value":         (loss_pg_value_baseline,  True, False),
    "vpg_entropy":       (loss_vanilla_pg,         False, True),
    "vpg_value_entropy": (loss_pg_value_baseline,   True, True),
}

ABLATION_ENTROPY_BETAS = [0.001, 0.01, 0.05, 0.1]
ABLATION_ENTROPY_SCHEDS = ["constant", "linear", "cosine"]


def build_method_list(K_values: list, run_ablations: bool) -> list:
    """Returns [(method_name, loss_fn, use_value_net, use_entropy, cfg_overrides)]."""
    methods = []

    for name, (loss_fn, use_v, use_ent) in METHODS_CORE.items():
        methods.append((name, loss_fn, use_v, use_ent, {}))

    for K in K_values:
        methods.append((
            f"rloo_K{K}",
            loss_rloo,
            False,
            False,
            {"episodes_per_update": K},
        ))

    if run_ablations:
        for beta in ABLATION_ENTROPY_BETAS:
            name = f"vpg_entropy_beta{beta}"
            methods.append((name, loss_vanilla_pg, False, True,
                            {"entropy_beta": beta}))

        for sched in ABLATION_ENTROPY_SCHEDS:
            name = f"vpg_entropy_sched_{sched}"
            methods.append((name, loss_vanilla_pg, False, True,
                            {"entropy_schedule": sched}))

    return methods



def _result_path(method_name: str, seed: int) -> str:
    safe = (method_name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", ""))
    return os.path.join(ROOT, "results", "raw", f"{safe}_seed{seed}.json")



def run_single(args_tuple):
    idx, total, method_name, loss_fn, use_v, use_ent, cfg = args_tuple

    path = _result_path(method_name, cfg.seed)
    if os.path.exists(path):
        print(f"  [skip {idx}/{total}] {method_name} seed={cfg.seed}", flush=True)
        return None

    print(f"  [{idx}/{total}] Running {method_name} seed={cfg.seed}...", flush=True)
    try:
        _, history = train(loss_fn, cfg, use_v, use_ent, method_name)
        return history
    except Exception:
        print(f"  ERROR [{idx}/{total}] {method_name} seed={cfg.seed}:\n"
              + traceback.format_exc(), flush=True)
        return None



def run_pg_experiments(cfg_base: Config, methods: list, n_seeds: int, n_cpus: int):
    tasks = []
    for method_name, loss_fn, use_v, use_ent, overrides in methods:
        for s in range(n_seeds):
            cfg = copy.copy(cfg_base)
            cfg.seed = cfg_base.seed + s
            for k, v in overrides.items():
                setattr(cfg, k, v)
            tasks.append((
                len(tasks) + 1, len(methods) * n_seeds,
                method_name, loss_fn, use_v, use_ent, cfg,
            ))

    print(f"\n{'='*60}")
    print(f"PART A — Policy Gradient  ({len(tasks)} runs, {n_cpus} worker(s))")
    print(f"{'='*60}")

    if n_cpus == 1:
        for task in tasks:
            run_single(task)
    else:
        with Pool(n_cpus) as pool:
            pool.map(run_single, tasks)



def run_bc_experiments(cfg_base: Config):
    print(f"\n{'='*60}")
    print("PART B — Behaviour Cloning")
    print(f"{'='*60}")

    expert_policy = _get_expert(cfg_base)

    print("\nVerifying expert policy (100 eval episodes)...")
    mean_r, std_r = evaluate(expert_policy, cfg_base, n_episodes=100)
    print(f"  Expert reward: {mean_r:.1f} ± {std_r:.1f}")
    if mean_r < 490:
        print(f"  WARNING: expert mean reward {mean_r:.1f} < 490. "
              "BC results may be weak.")

    bc_results = bc_failure_experiments(expert_policy, cfg_base)

    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
    bc_path = os.path.join(ROOT, "results", "raw", "bc_results.json")
    with open(bc_path, "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in bc_results.items()},
            f, indent=2,
        )
    print(f"\nBC results saved to {bc_path}")


def _get_expert(cfg_base: Config) -> PolicyNetwork:
    """Get or train expert for BC"""
    grouped = load_results()

    best_record = None
    best_reward = -float("inf")
    for method_name, records in grouped.items():
        if "vpg_value_entropy" not in method_name:
            continue
        for rec in records:
            h = rec["history"]
            rewards = h.get("episode_rewards", [])
            if not rewards:
                continue
            mean_r = float(np.mean(rewards[-100:]))
            if mean_r > best_reward:
                best_reward = mean_r
                best_record = rec

    if best_record is not None:
        print(f"\nBest expert: seed={best_record['config']['seed']}  "
              f"mean_last100={best_reward:.1f}")
        cfg = Config(**best_record["config"])
    else:
        print("\nNo saved vpg_value_entropy results. Training fresh...")
        cfg = copy.copy(cfg_base)

    policy, _ = train(
        loss_pg_value_baseline, cfg,
        use_value_net=True, use_entropy=True,
        method_name="vpg_value_entropy",
    )
    return policy



def print_summary():
    grouped = load_results()
    if not grouped:
        print("No results found in results/raw/.")
        return

    summary = {}
    rows = []
    for method_name, records in sorted(grouped.items()):
        histories = [rec["history"] for rec in records]
        mean_rewards = [
            float(np.mean(h["episode_rewards"][-100:])) if h.get("episode_rewards") else 0.0
            for h in histories
        ]
        solved = [bool(h.get("solved", False)) for h in histories]
        wall_times = [float(h.get("wall_time_total", 0.0)) for h in histories]

        summary[method_name] = {
            "mean_reward":      float(np.mean(mean_rewards)),
            "std_reward":       float(np.std(mean_rewards)),
            "solved_fraction":  float(sum(solved) / len(solved)),
            "mean_wall_time_s": float(np.mean(wall_times)),
            "n_seeds":          len(records),
        }
        rows.append((method_name, summary[method_name]))

    print(f"\n{'='*85}")
    print(f"{'Method':<35} | {'Mean R':>8} | {'Std':>6} | {'Solved':>7} | {'Time(s)':>8} | Seeds")
    print(f"{'-'*85}")
    for name, s in rows:
        print(f"{name:<35} | {s['mean_reward']:>8.1f} | {s['std_reward']:>6.1f} | "
              f"{s['solved_fraction']:>6.0%} | {s['mean_wall_time_s']:>8.1f} | {s['n_seeds']}")
    print(f"{'='*85}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    summary_path = os.path.join(ROOT, "results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")



def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick",   action="store_true")
    mode.add_argument("--full",    action="store_true")
    mode.add_argument("--bc-only", action="store_true")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        n_seeds, max_episodes, K_values = 1, 500, [8]
        run_ablations, n_cpus = False, 1
    elif args.full:
        n_seeds, max_episodes, K_values = 5, 1500, [4, 8, 16]
        run_ablations, n_cpus = True, min(4, cpu_count())
    else:  # --bc-only
        n_seeds, max_episodes, K_values = 1, 1500, [8]
        run_ablations, n_cpus = False, 1

    cfg_base = Config(seed=args.seed, max_episodes=max_episodes)

    if not args.bc_only:
        methods = build_method_list(K_values, run_ablations)
        run_pg_experiments(cfg_base, methods, n_seeds, n_cpus)

    run_bc_experiments(cfg_base)
    print_summary()


if __name__ == "__main__":
    # spawn required on macOS to avoid PyTorch + fork deadlocks
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
