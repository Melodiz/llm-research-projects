"""HW1 figures — run `python plotting.py` or `python plotting.py --figure N`."""

import argparse
import json
import math
import os
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator


_STYLES = ["seaborn-v0_8-paper", "seaborn-paper"]
for _s in _STYLES:
    try:
        plt.style.use(_s)
        break
    except OSError:
        continue

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       100,
})

FIG_SINGLE = (8, 5)
FIG_WIDE   = (12, 5)
DPI        = 300
SOLVE_Y    = 475.0
H_MAX      = math.log(2)

COLORS = {
    "vpg":               "#4878D0",
    "vpg_avg":           "#EE854A",
    "vpg_value":         "#6ACC65",
    "rloo_K4":           "#D65F5F",
    "rloo_K8":           "#956CB4",
    "rloo_K16":          "#8C613C",
    "vpg_entropy":       "#DC7EC0",
    "vpg_value_entropy": "#797979",
    "solve":             "#444444",
}

LABELS = {
    "vpg":               "VPG",
    "vpg_avg":           "VPG + Avg Baseline",
    "vpg_value":         "VPG + Value Baseline",
    "rloo_K4":           "RLOO K=4",
    "rloo_K8":           "RLOO K=8",
    "rloo_K16":          "RLOO K=16",
    "vpg_entropy":       "VPG + Entropy",
    "vpg_value_entropy": "VPG + Value + Entropy",
}



def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def load_bc(results_dir: str) -> dict:
    path = os.path.join(results_dir, "bc_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_grouped(results_dir: str) -> dict:
    from cartpole_pg import load_results
    return load_results(results_dir)


def get_histories(grouped: dict, key: str) -> list:
    records = grouped.get(key, [])
    return [r["history"] for r in records]


def warn_skip(fig_name: str, reason: str) -> bool:
    print(f"  [SKIP] {fig_name}: {reason}")
    return False


def save_fig(fig: plt.Figure, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _mean_std_across_seeds(histories: list, key: str, window: int = 1) -> tuple:
    """Smooth + align to shortest seed → (mean, std, x)."""
    arrays = [smooth(np.array(h[key], dtype=float), window) for h in histories
              if key in h and len(h[key]) > 0]
    if not arrays:
        return None, None, None
    min_len = min(len(a) for a in arrays)
    arrays = np.array([a[:min_len] for a in arrays])
    return arrays.mean(axis=0), arrays.std(axis=0), np.arange(min_len)


def _mean_std_padded(histories: list, key: str, window: int = 1) -> tuple:
    """Like _mean_std_across_seeds but pads short seeds to max length."""
    arrays = [smooth(np.array(h[key], dtype=float), window) for h in histories
              if key in h and len(h[key]) > 0]
    if not arrays:
        return None, None, None
    max_len = max(len(a) for a in arrays)
    padded = np.empty((len(arrays), max_len))
    for i, a in enumerate(arrays):
        padded[i, :len(a)] = a
        if len(a) < max_len:
            padded[i, len(a):] = a[-1]
    return padded.mean(axis=0), padded.std(axis=0), np.arange(max_len)



def fig1_learning_curves(grouped: dict, out_dir: str):
    name = "fig1_learning_curves"
    core_methods = ["vpg", "vpg_avg", "vpg_value", "rloo_K8",
                    "vpg_entropy", "vpg_value_entropy"]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    plotted = 0

    for key in core_methods:
        histories = get_histories(grouped, key)
        if not histories:
            continue
        mean, std, x = _mean_std_padded(histories, "episode_rewards", window=50)
        if mean is None:
            continue
        color = COLORS.get(key, "#333333")
        label = LABELS.get(key, key)
        ax.plot(x, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        plotted += 1

    if plotted == 0:
        warn_skip(name, "no episode_rewards data found")
        plt.close(fig)
        return

    ax.axhline(SOLVE_Y, color=COLORS["solve"], linestyle="--",
               linewidth=1, alpha=0.7, label="Solve threshold (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig2_grad_variance(grouped: dict, out_dir: str):
    name = "fig2_grad_variance"
    methods = ["vpg", "vpg_avg", "vpg_value", "rloo_K8"]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    plotted = 0

    for key in methods:
        histories = get_histories(grouped, key)
        if not histories:
            continue
        mean, std, x = _mean_std_across_seeds(histories, "grad_variances", window=20)
        if mean is None or np.all(mean == 0):
            continue
        color = COLORS.get(key, "#333333")
        label = LABELS.get(key, key)
        ax.plot(x, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, np.maximum(mean - std, 1e-30),
                        mean + std, color=color, alpha=0.15)
        plotted += 1

    if plotted == 0:
        warn_skip(name, "no grad_variances data found")
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Gradient Variance (log scale)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig3_entropy(grouped: dict, out_dir: str):
    name = "fig3_entropy"

    beta_methods = {
        k: v for k, v in grouped.items()
        if k.startswith("vpg_entropy_beta")
    }
    sched_methods = {
        k: v for k, v in grouped.items()
        if k.startswith("vpg_entropy_sched_")
    }

    if len(beta_methods) < 2 and len(sched_methods) < 2:
        warn_skip(name, "need >=2 beta ablation variants and >=2 schedule variants "
                  "(run with --full to generate them)")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE, sharey=True)

    def _plot_panel(ax, method_dict, title_suffix):
        ax.axhline(H_MAX, color=COLORS["solve"], linestyle="--",
                   linewidth=1, alpha=0.7, label=f"H_max = {H_MAX:.3f}")
        base = get_histories(grouped, "vpg_entropy")
        if base:
            mean, _, x = _mean_std_across_seeds(base, "entropies", window=10)
            if mean is not None and not np.all(mean == 0):
                ax.plot(x, mean, color=COLORS["vpg_entropy"],
                        label="VPG+Entropy (base)", linewidth=1.2, linestyle="--")

        for key, records in sorted(method_dict.items()):
            histories = [r["history"] for r in records]
            mean, std, x = _mean_std_across_seeds(histories, "entropies", window=10)
            if mean is None or np.all(mean == 0):
                continue
            label = key.replace("vpg_entropy_beta", "β=").replace("vpg_entropy_sched_", "")
            ax.plot(x, mean, label=label, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Policy Entropy")
        ax.legend(frameon=False)

    _plot_panel(axes[0], beta_methods, "β sweep")
    _plot_panel(axes[1], sched_methods, "schedule sweep")

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig4_rloo_ablation(grouped: dict, out_dir: str):
    name = "fig4_rloo_ablation"
    K_values = [4, 8, 16]
    keys = [f"rloo_K{K}" for K in K_values]

    available = [(K, k) for K, k in zip(K_values, keys) if k in grouped]
    if not available:
        warn_skip(name, "no rloo_K* results found")
        return

    means, stds, labels_bar, colors_bar = [], [], [], []
    for K, key in available:
        histories = get_histories(grouped, key)
        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        if not rewards:
            continue
        means.append(np.mean(rewards))
        stds.append(np.std(rewards) if len(rewards) > 1 else 0.0)
        labels_bar.append(f"K={K}")
        colors_bar.append(COLORS.get(key, "#555555"))

    if not means:
        warn_skip(name, "no episode_rewards data in rloo results")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(len(means))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4,
                  color=colors_bar, width=0.5, alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar)
    ax.set_xlabel("K (trajectories per update)")
    ax.set_ylabel("Mean Reward (last 100 episodes)")
    ax.axhline(SOLVE_Y, color=COLORS["solve"], linestyle="--",
               linewidth=1, alpha=0.7)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig5_bc_dataset_size(bc: dict, out_dir: str):
    name = "fig5_bc_dataset_size"
    if not bc:
        warn_skip(name, "bc_results.json not found")
        return

    size_entries = {
        int(k.replace("size_", "")): v
        for k, v in bc.items()
        if k.startswith("size_")
    }
    if not size_entries:
        warn_skip(name, "no size_* keys in bc_results.json")
        return

    sizes = sorted(size_entries)
    means = [size_entries[s]["mean"] for s in sizes]
    stds  = [size_entries[s]["std"]  for s in sizes]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.errorbar(sizes, means, yerr=stds, fmt="o-", capsize=4,
                color=COLORS["vpg_value"], linewidth=1.5, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Expert Transitions (log scale)")
    ax.set_ylabel("Mean Eval Reward")
    ax.axhline(SOLVE_Y, color=COLORS["solve"], linestyle="--",
               linewidth=1, alpha=0.7, label="Solve threshold (475)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



_BC_CACHE = {}


def _get_expert_and_bc(grouped: dict, n_transitions: int, caller: str):
    """Return (expert, bc_policy) or (None, None). Cached across figures."""
    cache_key = n_transitions
    if cache_key in _BC_CACHE:
        return _BC_CACHE[cache_key]

    records = grouped.get("vpg_value_entropy", [])
    if not records:
        warn_skip(caller, "no vpg_value_entropy results found")
        _BC_CACHE[cache_key] = (None, None)
        return None, None

    best_rec = max(records,
                   key=lambda r: np.mean(r["history"].get("episode_rewards", [-999])[-100:]))
    cfg_dict = best_rec["config"]

    print(f"  [{caller}] Retraining expert (seed={cfg_dict['seed']}, "
          f"episodes={cfg_dict['max_episodes']})...")

    try:
        from cartpole_pg import (Config, train, train_bc,
                                  collect_expert_data,
                                  loss_pg_value_baseline)

        cfg = Config(**cfg_dict)
        expert, _ = train(loss_pg_value_baseline, cfg,
                          use_value_net=True, use_entropy=True,
                          method_name="vpg_value_entropy")

        states, actions = collect_expert_data(expert, cfg, n_episodes=100)
        bc_policy, _ = train_bc(states, actions, cfg,
                                subset_size=n_transitions)
        print(f"  [{caller}] BC trained on {n_transitions} transitions")

    except Exception as e:
        warn_skip(caller, f"training failed: {e}")
        _BC_CACHE[cache_key] = (None, None)
        return None, None

    _BC_CACHE[cache_key] = (expert, bc_policy)
    return expert, bc_policy


def _rollout_states(policy_net, env, seed=0):
    state, _ = env.reset(seed=seed)
    states, angles = [], []
    done = False
    with torch.no_grad():
        while not done:
            states.append(state.copy())
            angles.append(float(state[2]))
            state_t = torch.FloatTensor(state).unsqueeze(0)
            dist = policy_net(state_t)
            action = dist.probs.argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(states), angles



def fig6_bc_state_dist(grouped: dict, bc: dict, out_dir: str):
    name = "fig6_bc_state_dist"

    expert, bc_weak = _get_expert_and_bc(grouped, n_transitions=500,
                                          caller="fig6")
    if expert is None:
        return

    env = gym.make("CartPole-v1")
    expert_visited, bc_visited = [], []

    for ep in range(50):
        s_exp, _ = _rollout_states(expert, env, seed=100 + ep)
        expert_visited.append(s_exp)
        s_bc, _ = _rollout_states(bc_weak, env, seed=100 + ep)
        bc_visited.append(s_bc)

    env.close()

    expert_all = np.concatenate(expert_visited)
    bc_all = np.concatenate(bc_visited)

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

    for ax, col, xlabel in [
        (axes[0], 0, "Cart Position"),
        (axes[1], 2, "Pole Angle (rad)"),
    ]:
        ax.hist(expert_all[:, col], bins=60, density=True, alpha=0.55,
                color=COLORS["vpg_value_entropy"], label="Expert")
        ax.hist(bc_all[:, col], bins=60, density=True, alpha=0.55,
                color=COLORS["vpg_entropy"], label="BC (500 trans.)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    fig.suptitle("State visitation: Expert vs BC trained on 500 transitions",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig6b_bc_state_dist(bc: dict, out_dir: str):
    name = "fig6b_bc_state_dist"
    if not bc or "state_dist" not in bc:
        warn_skip(name, "state_dist key missing from bc_results.json")
        return

    sd = bc["state_dist"]
    required = ["expert_cart_pos_std", "bc_cart_pos_std",
                "expert_pole_angle_std", "bc_pole_angle_std"]
    if not all(k in sd for k in required):
        warn_skip(name, "incomplete state_dist data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, dim, key_exp, key_bc, xlabel in [
        (axes[0], "Cart Position", "expert_cart_pos_std", "bc_cart_pos_std", "Cart Position"),
        (axes[1], "Pole Angle",   "expert_pole_angle_std", "bc_pole_angle_std", "Pole Angle (rad)"),
    ]:
        vals   = [sd[key_exp], sd[key_bc]]
        colors = [COLORS["vpg_value_entropy"], COLORS["vpg_entropy"]]
        bars   = ax.bar(["Expert", "BC Clone"], vals, color=colors, width=0.4, alpha=0.85)
        ax.set_ylabel("State Visitation Std")
        ax.set_xlabel(xlabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig7_bc_failure_episode(grouped: dict, bc: dict, out_dir: str):
    name = "fig7_bc_failure_episode"

    expert, bc_weak = _get_expert_and_bc(grouped, n_transitions=500,
                                          caller="fig7")
    if expert is None:
        return

    env = gym.make("CartPole-v1")
    FAIL_ANGLE = 0.2095

    _, expert_angles = _rollout_states(expert, env, seed=42)
    _, bc_angles = _rollout_states(bc_weak, env, seed=42)
    env.close()

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(expert_angles, color=COLORS["vpg_value_entropy"],
            label="Expert policy", linewidth=1.5)
    ax.plot(bc_angles, color=COLORS["vpg_entropy"],
            label="BC (500 trans.)", linewidth=1.5, linestyle="--")

    ax.axhline( FAIL_ANGLE, color="#cc0000", linestyle=":", linewidth=1,
               alpha=0.8, label="Failure threshold (±0.21 rad)")
    ax.axhline(-FAIL_ANGLE, color="#cc0000", linestyle=":", linewidth=1, alpha=0.8)

    fail_step = next((i for i, a in enumerate(bc_angles) if abs(a) >= FAIL_ANGLE), None)
    if fail_step is not None:
        ax.axvline(fail_step, color="#cc4400", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"BC failure (t={fail_step})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pole Angle (rad)")
    ax.set_title("Single episode: Expert vs BC trained on 500 transitions")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig7b_bc_failure_episode(grouped: dict, bc: dict, out_dir: str):
    name = "fig7b_bc_failure_episode"

    records = grouped.get("vpg_value_entropy", [])
    if not records:
        warn_skip(name, "no vpg_value_entropy results found (need --full or --quick run)")
        return

    best_rec = max(records, key=lambda r: np.mean(r["history"].get("episode_rewards", [-999])[-100:]))
    cfg_dict = best_rec["config"]

    print(f"  [fig7b] Retraining expert (seed={cfg_dict['seed']}, "
          f"episodes={cfg_dict['max_episodes']})...")

    try:
        from cartpole_pg import (Config, train, train_bc,
                                  collect_expert_data,
                                  loss_pg_value_baseline)

        cfg = Config(**cfg_dict)
        expert, _ = train(loss_pg_value_baseline, cfg,
                          use_value_net=True, use_entropy=True,
                          method_name="vpg_value_entropy")

        states, actions = collect_expert_data(expert, cfg, n_episodes=50)
        bc_policy, _ = train_bc(states, actions, cfg)

    except Exception as e:
        warn_skip(name, f"expert/BC training failed: {e}")
        return

    env = gym.make("CartPole-v1")
    FAIL_ANGLE = 0.2095

    _, expert_angles = _rollout_states(expert, env, seed=42)
    _, bc_angles = _rollout_states(bc_policy, env, seed=42)
    env.close()

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(expert_angles, color=COLORS["vpg_value_entropy"],
            label="Expert policy", linewidth=1.5)
    ax.plot(bc_angles, color=COLORS["vpg_entropy"],
            label="BC clone (full data)", linewidth=1.5, linestyle="--")

    ax.axhline( FAIL_ANGLE, color="#cc0000", linestyle=":", linewidth=1,
               alpha=0.8, label="Failure threshold (±0.21 rad)")
    ax.axhline(-FAIL_ANGLE, color="#cc0000", linestyle=":", linewidth=1, alpha=0.8)

    fail_step = next((i for i, a in enumerate(bc_angles) if abs(a) >= FAIL_ANGLE), None)
    if fail_step is not None:
        ax.axvline(fail_step, color="#cc4400", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"BC enters unseen states (t={fail_step})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pole Angle (rad)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



def fig8_summary_table(grouped: dict, out_dir: str):
    name = "fig8_summary_table"
    if not grouped:
        warn_skip(name, "no results loaded")
        return

    col_headers = ["Method", "Mean Reward", "Std", "Solved", "Time (s)"]
    rows = []

    method_order = ["vpg", "vpg_avg", "vpg_value", "rloo_K4", "rloo_K8", "rloo_K16",
                    "vpg_entropy", "vpg_value_entropy"]
    present = set(grouped.keys())
    keys_to_show = [k for k in method_order if k in present]
    keys_to_show += [k for k in sorted(present) if k not in method_order]

    for key in keys_to_show:
        histories = get_histories(grouped, key)
        if not histories:
            continue
        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        solved  = [h.get("solved", False) for h in histories]
        times   = [h.get("wall_time_total", 0.0) for h in histories]

        mean_r = np.mean(rewards) if rewards else 0.0
        std_r  = np.std(rewards) if len(rewards) > 1 else 0.0
        solved_f = f"{sum(solved)}/{len(solved)}"
        mean_t = np.mean(times) if times else 0.0

        rows.append([
            LABELS.get(key, key),
            f"{mean_r:.1f}",
            f"{std_r:.1f}",
            solved_f,
            f"{mean_t:.0f}",
        ])

    if not rows:
        warn_skip(name, "no data to show")
        return

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        color = "#f5f5f5" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_headers)):
            tbl[(i, j)].set_facecolor(color)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{name}.png"))



ALL_FIGURES = {
    1:  ("fig1_learning_curves",     lambda g, bc, d: fig1_learning_curves(g, d)),
    2:  ("fig2_grad_variance",       lambda g, bc, d: fig2_grad_variance(g, d)),
    3:  ("fig3_entropy",             lambda g, bc, d: fig3_entropy(g, d)),
    4:  ("fig4_rloo_ablation",       lambda g, bc, d: fig4_rloo_ablation(g, d)),
    5:  ("fig5_bc_dataset_size",     lambda g, bc, d: fig5_bc_dataset_size(bc, d)),
    6:  ("fig6_bc_state_dist",       lambda g, bc, d: fig6_bc_state_dist(g, bc, d)),
    7:  ("fig7_bc_failure_episode",  lambda g, bc, d: fig7_bc_failure_episode(g, bc, d)),
    8:  ("fig8_summary_table",       lambda g, bc, d: fig8_summary_table(g, d)),
    61: ("fig6b_bc_state_dist",      lambda g, bc, d: fig6b_bc_state_dist(bc, d)),
    71: ("fig7b_bc_failure_episode", lambda g, bc, d: fig7b_bc_failure_episode(g, bc, d)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, default=None)
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "raw"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    grouped = load_grouped(args.results_dir)
    bc      = load_bc(args.results_dir)
    print(f"  Methods found: {sorted(grouped.keys())}")
    print(f"  BC results:    {'yes' if bc else 'no'}")

    figures_to_run = (
        {args.figure: ALL_FIGURES[args.figure]}
        if args.figure is not None
        else ALL_FIGURES
    )

    for num, (label, fn) in figures_to_run.items():
        print(f"\nGenerating Figure {num}: {label}")
        try:
            fn(grouped, bc, args.out_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR in Figure {num}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
