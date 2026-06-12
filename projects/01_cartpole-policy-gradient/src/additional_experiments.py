"""Additional experiments: more seeds, RLOO curve, BC noise sweep."""

import argparse
import copy
import json
import os
import traceback
import time
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cartpole_pg import (
    Config,
    PolicyNetwork,
    train,
    evaluate,
    train_bc,
    collect_expert_data,
    load_results,
    loss_vanilla_pg,
    loss_pg_avg_baseline,
    loss_pg_value_baseline,
    loss_rloo,
)



def _result_path(method_name: str, seed: int) -> str:
    safe = (method_name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", ""))
    return os.path.join(ROOT, "results", "raw", f"{safe}_seed{seed}.json")


def _run_single(args_tuple):
    idx, total, method_name, loss_fn, use_v, use_ent, cfg = args_tuple
    path = _result_path(method_name, cfg.seed)
    if os.path.exists(path):
        print(f"  [skip {idx}/{total}] {method_name} seed={cfg.seed}", flush=True)
        return None
    print(f"  [{idx}/{total}] {method_name} seed={cfg.seed}...", flush=True)
    try:
        _, history = train(loss_fn, cfg, use_v, use_ent, method_name)
        return history
    except Exception:
        print(f"  ERROR [{idx}/{total}] {method_name} seed={cfg.seed}:\n"
              + traceback.format_exc(), flush=True)
        return None


def _run_tasks(tasks: list, n_cpus: int, label: str, est_min_per_run: float = 2.0):
    new_count = sum(1 for t in tasks if not os.path.exists(_result_path(t[2], t[6].seed)))
    est_minutes = est_min_per_run * new_count / max(n_cpus, 1)
    print(f"\n{'='*60}")
    print(f"{label}  ({len(tasks)} total, ~{new_count} new, {n_cpus} workers)")
    print(f"Estimated time: ~{est_minutes:.0f} min")
    print(f"{'='*60}")

    if n_cpus == 1:
        for task in tasks:
            _run_single(task)
    else:
        with Pool(n_cpus) as pool:
            pool.map(_run_single, tasks)


def _get_expert(cfg_base: Config) -> PolicyNetwork:
    """Re-train best saved vpg_value_entropy seed, or train fresh."""
    grouped = load_results()
    best_record = None
    best_reward = -float("inf")
    for method_name, records in grouped.items():
        if "vpg_value_entropy" not in method_name:
            continue
        for rec in records:
            rewards = rec["history"].get("episode_rewards", [])
            if not rewards:
                continue
            mean_r = float(np.mean(rewards[-100:]))
            if mean_r > best_reward:
                best_reward = mean_r
                best_record = rec

    if best_record is not None:
        print(f"  Best expert: seed={best_record['config']['seed']}  "
              f"mean_last100={best_reward:.1f}")
        cfg = Config(**best_record["config"])
    else:
        print("  No saved vpg_value_entropy results. Training fresh...")
        cfg = copy.copy(cfg_base)

    policy, _ = train(
        loss_pg_value_baseline, cfg,
        use_value_net=True, use_entropy=True,
        method_name="vpg_value_entropy",
    )
    return policy



EXTRA_SEED_METHODS = [
    ("vpg",               loss_vanilla_pg,        False, False, {}),
    ("vpg_avg",           loss_pg_avg_baseline,   False, False, {}),
    ("vpg_value",         loss_pg_value_baseline,  True, False, {}),
    ("vpg_entropy",       loss_vanilla_pg,         False, True,
     {"entropy_beta": 0.01, "entropy_schedule": "linear"}),
    ("vpg_value_entropy", loss_pg_value_baseline,   True, True,
     {"entropy_beta": 0.01, "entropy_schedule": "linear"}),
    ("rloo_K4",           loss_rloo,              False, False, {"episodes_per_update": 4}),
    ("rloo_K8",           loss_rloo,              False, False, {"episodes_per_update": 8}),
    ("rloo_K16",          loss_rloo,              False, False, {"episodes_per_update": 16}),
]


def run_extra_seeds(cfg_base: Config, seed_start: int, seed_end: int, n_cpus: int):
    tasks = []
    for name, loss_fn, use_v, use_ent, overrides in EXTRA_SEED_METHODS:
        for s in range(seed_start, seed_end + 1):
            cfg = copy.copy(cfg_base)
            cfg.seed = s
            for k, v in overrides.items():
                setattr(cfg, k, v)
            tasks.append((len(tasks) + 1, 0, name, loss_fn, use_v, use_ent, cfg))

    # fill in total count
    tasks = [(idx, len(tasks), name, fn, uv, ue, c)
             for idx, _, name, fn, uv, ue, c in tasks]

    _run_tasks(tasks, n_cpus, "Extra seeds (all methods)")



RLOO_K_VALUES = [2, 3, 4, 6, 8, 12, 16, 24]


def run_rloo_curve(cfg_base: Config, n_cpus: int):
    tasks = []
    for K in RLOO_K_VALUES:
        for s in range(42, 57):  # seeds 42-56
            cfg = copy.copy(cfg_base)
            cfg.seed = s
            cfg.episodes_per_update = K
            name = f"rloo_K{K}"
            tasks.append((len(tasks) + 1, 0, name, loss_rloo, False, False, cfg))

    tasks = [(idx, len(tasks), name, fn, uv, ue, c)
             for idx, _, name, fn, uv, ue, c in tasks]

    _run_tasks(tasks, n_cpus, "RLOO K curve (K=2..24, 15 seeds)")



NOISE_FRACS = [round(x, 2) for x in np.arange(0, 1.05, 0.05).tolist()]
TRANSITION_ZONE = {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}


def run_bc_noise_full(cfg_base: Config):
    print(f"\n{'='*60}")
    print("BC noise sweep (21 noise levels, multi-seed transition zone)")
    print(f"{'='*60}")

    out_path = os.path.join(ROOT, "results", "raw", "bc_noise_full.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Resuming from {out_path} ({len(results)} noise levels saved)")
    else:
        results = {}

    print("  Training expert...")
    expert = _get_expert(cfg_base)

    print("  Collecting expert data...")
    states, actions = collect_expert_data(expert, cfg_base, n_episodes=100)
    states = states[:10000]
    actions = actions[:10000]
    print(f"  Dataset: {len(states)} transitions")

    # Count total runs
    run_list = []
    for nf in NOISE_FRACS:
        seeds = list(range(42, 47)) if nf in TRANSITION_ZONE else [42]
        for seed in seeds:
            nf_key = f"{nf:.2f}"
            if nf_key in results and str(seed) in results[nf_key]:
                continue
            run_list.append((nf, seed))

    total = len(run_list)
    est = total * 5 / 60  # ~5s each
    print(f"  {total} BC trains to run (~{est:.1f} min)")

    for i, (nf, seed) in enumerate(run_list):
        np.random.seed(seed)
        noisy = actions.copy()
        if nf > 0:
            mask = np.random.random(len(noisy)) < nf
            noisy[mask] = 1 - noisy[mask]

        bc_policy, _ = train_bc(states, noisy, cfg_base, subset_size=10000)
        mean_r, std_r = evaluate(bc_policy, cfg_base, n_episodes=20)

        nf_key = f"{nf:.2f}"
        if nf_key not in results:
            results[nf_key] = {}
        results[nf_key][str(seed)] = {
            "mean_reward": float(mean_r),
            "std_reward": float(std_r),
        }

        print(f"  [{i+1}/{total}] noise={nf*100:5.1f}% seed={seed} -> {mean_r:.1f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")



def run_entropy_extra_seeds(cfg_base: Config, seed_start: int, seed_end: int, n_cpus: int):
    tasks = []
    for s in range(seed_start, seed_end + 1):
        cfg = copy.copy(cfg_base)
        cfg.seed = s
        cfg.entropy_beta = 0.01
        cfg.entropy_schedule = "linear"
        tasks.append((len(tasks) + 1, 0,
                       "vpg_entropy", loss_vanilla_pg, False, True, cfg))

    tasks = [(idx, len(tasks), name, fn, uv, ue, c)
             for idx, _, name, fn, uv, ue, c in tasks]

    _run_tasks(tasks, n_cpus, "Entropy extra seeds (vpg_entropy, linear)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-seeds", action="store_true")
    parser.add_argument("--rloo-curve", action="store_true")
    parser.add_argument("--bc-noise-full", action="store_true")
    parser.add_argument("--entropy-extra-seeds", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.extra_seeds = True
        args.rloo_curve = True
        args.bc_noise_full = True
        args.entropy_extra_seeds = True

    if not any([args.extra_seeds, args.rloo_curve, args.bc_noise_full,
                args.entropy_extra_seeds]):
        parser.print_help()
        return

    cfg_base = Config(max_episodes=1500)
    n_cpus = min(4, cpu_count())

    if args.extra_seeds:
        run_extra_seeds(cfg_base, seed_start=47, seed_end=56, n_cpus=n_cpus)

    if args.rloo_curve:
        run_rloo_curve(cfg_base, n_cpus=n_cpus)

    if args.entropy_extra_seeds:
        run_entropy_extra_seeds(cfg_base, seed_start=47, seed_end=56, n_cpus=n_cpus)

    if args.bc_noise_full:
        run_bc_noise_full(cfg_base)

    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
