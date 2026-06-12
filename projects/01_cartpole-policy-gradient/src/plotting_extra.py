#!/usr/bin/env python3
"""Extra figures from additional experiments (fig9, fig10)."""

import json
import os
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting import (
    COLORS, FIG_SINGLE, DPI, SOLVE_Y,
    load_grouped, get_histories,
)



def mean_last100(history):
    return float(np.mean(history["episode_rewards"][-100:]))


def solve_episode(history, threshold=475.0, window=5):
    evals = history.get("eval_rewards", [])
    if len(evals) < window:
        return None
    for i in range(len(evals) - window + 1):
        if all(evals[i + j]["mean"] >= threshold for j in range(window)):
            return evals[i]["episode"]
    return None


def _sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b



def fig9_rloo_curve(grouped, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(ROOT, "figures")
    import re

    k_data = {}
    for method, records in grouped.items():
        m = re.match(r"rloo_K(\d+)$", method)
        if not m:
            continue
        K = int(m.group(1))
        rewards = [mean_last100(r["history"]) for r in records]
        k_data[K] = np.array(rewards)

    if not k_data:
        print("  [SKIP] fig9: no RLOO results")
        return

    ks = sorted(k_data)
    means = [np.mean(k_data[K]) for K in ks]
    # 95% CI: mean ± 1.96 * SE
    cis = [1.96 * np.std(k_data[K]) / np.sqrt(len(k_data[K])) for K in ks]

    best_K = ks[int(np.argmax(means))]
    best_mean = max(means)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.errorbar(ks, means, yerr=cis, fmt="o-", color=COLORS["rloo_K4"],
                capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label="Mean reward (95% CI)")

    ax.axhline(SOLVE_Y, color=COLORS["solve"], linestyle="--",
               linewidth=1, alpha=0.7, label=f"Solve threshold ({SOLVE_Y:.0f})")

    # annotate optimal K
    ax.annotate(f"K={best_K}\n{best_mean:.0f}",
                xy=(best_K, best_mean), xytext=(best_K + 2, best_mean - 30),
                fontsize=10, fontweight="bold", color=COLORS["rloo_K4"],
                arrowprops=dict(arrowstyle="->", color=COLORS["rloo_K4"], lw=1.5))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    update_ticks = [2, 4, 8, 16, 24]
    ax2.set_xticks(update_ticks)
    ax2.set_xticklabels([f"{1500 // K}" for K in update_ticks])
    ax2.set_xlabel("Gradient updates (1500 / K)", fontsize=10)

    ax.set_xlabel("Group size K")
    ax.set_ylabel("Mean reward (last 100 episodes)")
    ax.set_title("RLOO: Reward vs Group Size K")
    ax.set_xticks(ks)
    ax.legend(loc="lower left")
    ax.set_ylim(100, 520)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "fig9_rloo_curve.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")



def fig10_bc_noise_curve(save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(ROOT, "figures")
    bc_path = os.path.join(ROOT, "results", "raw", "bc_noise_full.json")
    if not os.path.exists(bc_path):
        print("  [SKIP] fig10: bc_noise_full.json not found")
        return

    with open(bc_path) as f:
        raw = json.load(f)

    noise_fracs = []
    means = []
    ci_lo = []
    ci_hi = []

    for nf_str in sorted(raw.keys(), key=float):
        nf = float(nf_str)
        rewards = [v["mean_reward"] for v in raw[nf_str].values()]
        mu = np.mean(rewards)
        noise_fracs.append(nf * 100)  # percent
        means.append(mu)
        if len(rewards) > 1:
            se = 1.96 * np.std(rewards) / np.sqrt(len(rewards))
            ci_lo.append(mu - se)
            ci_hi.append(mu + se)
        else:
            ci_lo.append(mu)
            ci_hi.append(mu)

    noise_fracs = np.array(noise_fracs)
    means = np.array(means)
    ci_lo = np.array(ci_lo)
    ci_hi = np.array(ci_hi)

    from scipy.optimize import curve_fit
    nf_01 = noise_fracs / 100.0
    try:
        p0 = [-400, 15, 0.5, 450]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_sigmoid, nf_01, means, p0=p0, maxfev=10000)
        x_fit = np.linspace(0, 1, 200)
        y_fit = _sigmoid(x_fit, *popt)
        inflection = popt[2]  # x0
        fit_ok = True
    except RuntimeError:
        fit_ok = False

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    yerr_lo = means - ci_lo
    yerr_hi = ci_hi - means
    ax.errorbar(noise_fracs, means, yerr=[yerr_lo, yerr_hi],
                fmt="o", color=COLORS["vpg"], capsize=3, capthick=1.2,
                markersize=6, zorder=5, label="BC reward (95% CI)")

    if fit_ok:
        ax.plot(x_fit * 100, y_fit, "-", color=COLORS["rloo_K4"],
                linewidth=2, alpha=0.8, label="Sigmoid fit")
        ax.axvline(inflection * 100, color=COLORS["rloo_K4"], linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.annotate(f"transition at {inflection * 100:.0f}%",
                    xy=(inflection * 100, _sigmoid(inflection, *popt)),
                    xytext=(inflection * 100 + 8, 280),
                    fontsize=10, color=COLORS["rloo_K4"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["rloo_K4"], lw=1.2))

    ax.axhline(500, color=COLORS["vpg_value"], linestyle=":", linewidth=1,
               alpha=0.7, label="Expert (500)")
    ax.axhline(20, color=COLORS["solve"], linestyle=":", linewidth=1,
               alpha=0.7, label="Random (~20)")

    ax.set_xlabel("Expert label noise (%)")
    ax.set_ylabel("Mean reward")
    ax.set_title("BC Robustness to Expert Label Noise")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-10, 540)
    ax.legend(loc="center left")

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "fig10_bc_noise_curve.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")



def main():
    print("Generating extra figures...")
    grouped = load_grouped(os.path.join(ROOT, "results", "raw"))
    fig9_rloo_curve(grouped)
    fig10_bc_noise_curve()
    print("Done.")


if __name__ == "__main__":
    main()
