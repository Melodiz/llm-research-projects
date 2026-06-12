#!/usr/bin/env python3
"""Statistical tests for HW1 report claims."""

import json
import os
import re
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from cartpole_pg import load_results



def solve_episode(history, threshold=475.0, window=5):
    """First episode with `window` consecutive evals >= threshold."""
    evals = history.get("eval_rewards", [])
    if len(evals) < window:
        return None
    for i in range(len(evals) - window + 1):
        if all(evals[i + j]["mean"] >= threshold for j in range(window)):
            return evals[i]["episode"]
    return None


def mean_last100(history):
    return float(np.mean(history["episode_rewards"][-100:]))


def get_seed_matched(grouped, method_a, method_b, metric_fn=mean_last100):
    """Return paired arrays for seeds present in both methods."""
    seeds_a = {r["config"]["seed"]: metric_fn(r["history"])
               for r in grouped.get(method_a, [])}
    seeds_b = {r["config"]["seed"]: metric_fn(r["history"])
               for r in grouped.get(method_b, [])}
    common = sorted(set(seeds_a) & set(seeds_b))
    a = np.array([seeds_a[s] for s in common])
    b = np.array([seeds_b[s] for s in common])
    return a, b, common


def bootstrap_ci(data, stat_fn=np.mean, n_boot=10000, ci=95):
    rng = np.random.RandomState(0)
    data = np.asarray(data)
    stats_boot = np.array([stat_fn(rng.choice(data, len(data), replace=True))
                           for _ in range(n_boot)])
    lo = np.percentile(stats_boot, (100 - ci) / 2)
    hi = np.percentile(stats_boot, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def bootstrap_ci_diff(a, b, n_boot=10000, ci=95):
    rng = np.random.RandomState(0)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.array([
        np.mean(rng.choice(a, n, replace=True)) - np.mean(rng.choice(b, n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(diffs, (100 - ci) / 2)
    hi = np.percentile(diffs, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                     / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _fmt_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"



PAIRS = [
    ("vpg_value", "vpg", "Value baseline effect"),
    ("vpg_value_entropy", "vpg_value", "Entropy effect"),
    ("rloo_K4", "vpg", "RLOO vs vanilla"),
    ("vpg_avg", "vpg", "Avg baseline effect"),
]


def paired_comparisons(grouped):
    print("\n" + "=" * 60)
    print("PAIRED COMPARISONS")
    print("=" * 60)

    results = []
    for method_a, method_b, label in PAIRS:
        a, b, seeds = get_seed_matched(grouped, method_a, method_b)
        n = len(seeds)
        if n < 3:
            print(f"\n  {label}: skipped (only {n} matched seeds)")
            continue

        diff = float(np.mean(a) - np.mean(b))
        t_stat, t_p = stats.ttest_rel(a, b)

        # wilcoxon needs n >= 6 for meaningful result
        if n >= 6:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")

        ci_lo, ci_hi = bootstrap_ci_diff(a, b)
        d = cohens_d(a, b)

        rec = {
            "label": label,
            "method_a": method_a, "method_b": method_b,
            "n_seeds": n,
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "diff": diff,
            "t_stat": float(t_stat), "t_p": float(t_p),
            "w_stat": float(w_stat), "w_p": float(w_p),
            "boot_ci_95": [ci_lo, ci_hi],
            "cohens_d": d,
        }
        results.append(rec)

        print(f"\n  {label} ({method_a} vs {method_b}, n={n})")
        print(f"    mean_a={np.mean(a):.1f}  mean_b={np.mean(b):.1f}  diff={diff:+.1f}")
        print(f"    t-test: t={t_stat:.3f}  p={_fmt_p(t_p)}")
        if not np.isnan(w_p):
            print(f"    Wilcoxon: W={w_stat:.1f}  p={_fmt_p(w_p)}")
        print(f"    Bootstrap 95% CI of diff: [{ci_lo:.1f}, {ci_hi:.1f}]")
        print(f"    Cohen's d: {d:.3f}")

    return results



def rloo_curve_analysis(grouped):
    print("\n" + "=" * 60)
    print("RLOO K CURVE")
    print("=" * 60)

    k_data = {}
    for method, records in grouped.items():
        m = re.match(r"rloo_K(\d+)$", method)
        if not m:
            continue
        K = int(m.group(1))
        rewards = [mean_last100(r["history"]) for r in records]
        solve_count = sum(1 for r in records
                         if solve_episode(r["history"]) is not None)
        k_data[K] = {
            "rewards": rewards,
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "n_seeds": len(rewards),
            "solve_rate": solve_count / len(rewards),
        }

    if not k_data:
        print("  No RLOO results found.")
        return {}

    print(f"\n  {'K':>4}  {'mean':>7}  {'std':>7}  {'n':>3}  {'solve%':>6}")
    for K in sorted(k_data):
        d = k_data[K]
        print(f"  {K:4d}  {d['mean']:7.1f}  {d['std']:7.1f}  {d['n_seeds']:3d}  "
              f"{d['solve_rate']*100:5.1f}%")

    all_k = []
    all_r = []
    for K in sorted(k_data):
        for r in k_data[K]["rewards"]:
            all_k.append(K)
            all_r.append(r)

    rho, sp_p = stats.spearmanr(all_k, all_r)
    print(f"\n  Spearman (K vs reward): rho={rho:.3f}  p={_fmt_p(sp_p)}")

    best_K = max(k_data, key=lambda k: k_data[k]["mean"])
    print(f"  Optimal K: {best_K} (mean={k_data[best_K]['mean']:.1f})")

    if 8 in k_data and best_K != 8:
        ci = bootstrap_ci_diff(
            np.array(k_data[best_K]["rewards"]),
            np.array(k_data[8]["rewards"]),
        )
        print(f"  K={best_K} vs K=8 advantage 95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")

    result = {
        "per_K": {str(K): {k: v for k, v in d.items() if k != "rewards"}
                  for K, d in k_data.items()},
        "spearman_rho": float(rho),
        "spearman_p": float(sp_p),
        "optimal_K": best_K,
    }
    return result



ENTROPY_METHODS = ["vpg_entropy", "vpg_value_entropy"]
NON_ENTROPY_METHODS = ["vpg", "vpg_avg", "vpg_value"]


def solve_rate_analysis(grouped):
    print("\n" + "=" * 60)
    print("SOLVE RATE: ENTROPY vs NON-ENTROPY")
    print("=" * 60)

    ent_solved, ent_total = 0, 0
    for m in ENTROPY_METHODS:
        for r in grouped.get(m, []):
            ent_total += 1
            if solve_episode(r["history"]) is not None:
                ent_solved += 1

    non_solved, non_total = 0, 0
    for m in NON_ENTROPY_METHODS:
        for r in grouped.get(m, []):
            non_total += 1
            if solve_episode(r["history"]) is not None:
                non_solved += 1

    print(f"  Entropy methods:     {ent_solved}/{ent_total} solved")
    print(f"  Non-entropy methods: {non_solved}/{non_total} solved")

    table = np.array([
        [ent_solved, ent_total - ent_solved],
        [non_solved, non_total - non_solved],
    ])
    odds, p = stats.fisher_exact(table)
    print(f"  Fisher exact: odds_ratio={odds:.3f}  p={_fmt_p(p)}")

    return {
        "entropy_solved": ent_solved, "entropy_total": ent_total,
        "non_entropy_solved": non_solved, "non_entropy_total": non_total,
        "odds_ratio": float(odds), "p_value": float(p),
    }



def _sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def bc_noise_analysis():
    print("\n" + "=" * 60)
    print("BC NOISE CURVE")
    print("=" * 60)

    path = os.path.join(ROOT, "results", "raw", "bc_noise_full.json")
    if not os.path.exists(path):
        print("  bc_noise_full.json not found, skipping.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    noise_levels = []
    mean_rewards = []
    all_rewards = {}  # noise -> list of rewards (for bootstrap)

    for nf_str in sorted(raw.keys(), key=float):
        nf = float(nf_str)
        rewards = [v["mean_reward"] for v in raw[nf_str].values()]
        noise_levels.append(nf)
        mean_rewards.append(float(np.mean(rewards)))
        all_rewards[nf] = rewards

    noise_levels = np.array(noise_levels)
    mean_rewards = np.array(mean_rewards)

    print(f"  {len(noise_levels)} noise levels, rewards range "
          f"[{mean_rewards.min():.0f}, {mean_rewards.max():.0f}]")

    try:
        # fit descending sigmoid: L / (1 + exp(k*(x-x0))) + b
        # equivalently: L / (1 + exp(-(-k)*(x-x0))) + b with negative k
        p0 = [-400, 15, 0.5, 450]  # L negative (drop), steep, midpoint ~0.5
        popt, pcov = curve_fit(_sigmoid, noise_levels, mean_rewards, p0=p0,
                               maxfev=10000)
        L, k, x0, b = popt
        print(f"  Sigmoid fit: L={L:.1f}, k={k:.2f}, x0={x0:.3f}, b={b:.1f}")

        threshold_reward = 250.0
        # solve: 250 = L/(1+exp(-k*(x-x0))) + b  =>  x = x0 - ln(L/(250-b) - 1)/k
        if L != 0 and k != 0:
            ratio = L / (threshold_reward - b)
            if ratio > 1:
                threshold = x0 - np.log(ratio - 1) / k
            else:
                threshold = float("nan")
        else:
            threshold = float("nan")
        print(f"  Threshold (reward < 250): noise = {threshold:.3f}")

        rng = np.random.RandomState(42)
        thresholds = []
        for _ in range(2000):
            boot_means = []
            for nf in noise_levels:
                rews = all_rewards[nf]
                boot_means.append(float(np.mean(rng.choice(rews, len(rews), replace=True))))
            boot_means = np.array(boot_means)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bp, _ = curve_fit(_sigmoid, noise_levels, boot_means, p0=popt,
                                      maxfev=5000)
                bL, bk, bx0, bb = bp
                if bL != 0 and bk != 0:
                    r = bL / (threshold_reward - bb)
                    if r > 1:
                        thresholds.append(bx0 - np.log(r - 1) / bk)
            except (RuntimeError, ValueError):
                continue

        if thresholds:
            ci_lo = float(np.percentile(thresholds, 2.5))
            ci_hi = float(np.percentile(thresholds, 97.5))
            print(f"  Threshold 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
        else:
            ci_lo, ci_hi = float("nan"), float("nan")

        result = {
            "sigmoid_params": {"L": float(L), "k": float(k),
                               "x0": float(x0), "b": float(b)},
            "threshold_reward_250": float(threshold),
            "threshold_ci_95": [ci_lo, ci_hi],
            "noise_levels": noise_levels.tolist(),
            "mean_rewards": mean_rewards.tolist(),
        }

    except RuntimeError as e:
        print(f"  Sigmoid fit failed: {e}")
        result = {
            "error": str(e),
            "noise_levels": noise_levels.tolist(),
            "mean_rewards": mean_rewards.tolist(),
        }

    return result



GRAD_METHODS = ["vpg", "vpg_avg", "vpg_value", "vpg_value_entropy", "rloo_K4"]


def grad_variance_analysis(grouped):
    print("\n" + "=" * 60)
    print("GRADIENT VARIANCE (EARLY vs LATE)")
    print("=" * 60)

    results = {}
    for method in GRAD_METHODS:
        records = grouped.get(method, [])
        if not records:
            continue

        early_means = []
        late_means = []
        for r in records:
            gv = r["history"].get("grad_variances", [])
            if len(gv) < 20:
                continue
            early_means.append(float(np.mean(gv[:10])))
            late_means.append(float(np.mean(gv[-20:])))

        if len(early_means) < 3:
            print(f"\n  {method}: skipped ({len(early_means)} seeds with enough data)")
            continue

        early = np.array(early_means)
        late = np.array(late_means)

        t_stat, t_p = stats.ttest_rel(early, late)
        early_ci = bootstrap_ci(early)
        late_ci = bootstrap_ci(late)
        ratio = float(np.mean(early) / np.mean(late)) if np.mean(late) > 0 else float("inf")

        rec = {
            "n_seeds": len(early),
            "early_mean": float(np.mean(early)),
            "early_ci_95": list(early_ci),
            "late_mean": float(np.mean(late)),
            "late_ci_95": list(late_ci),
            "ratio_early_over_late": ratio,
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
        }
        results[method] = rec

        print(f"\n  {method} (n={len(early)})")
        print(f"    early: {np.mean(early):.2e}  CI [{early_ci[0]:.2e}, {early_ci[1]:.2e}]")
        print(f"    late:  {np.mean(late):.2e}  CI [{late_ci[0]:.2e}, {late_ci[1]:.2e}]")
        print(f"    ratio (early/late): {ratio:.2f}")
        print(f"    paired t: t={t_stat:.3f}  p={_fmt_p(t_p)}")

    return results



def main():
    grouped = load_results()
    print(f"Loaded {sum(len(v) for v in grouped.values())} records "
          f"across {len(grouped)} methods")

    paired = paired_comparisons(grouped)
    rloo = rloo_curve_analysis(grouped)
    solve = solve_rate_analysis(grouped)
    bc = bc_noise_analysis()
    grad = grad_variance_analysis(grouped)

    # save outputs
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    stat_tests = {
        "paired_comparisons": paired,
        "solve_rate": solve,
        "gradient_variance": grad,
    }
    with open(os.path.join(ROOT, "results", "statistical_tests.json"), "w") as f:
        json.dump(stat_tests, f, indent=2)
    print(f"\nSaved results/statistical_tests.json")

    if rloo:
        with open(os.path.join(ROOT, "results", "rloo_curve.json"), "w") as f:
            json.dump(rloo, f, indent=2)
        print(f"Saved results/rloo_curve.json")

    if bc:
        with open(os.path.join(ROOT, "results", "bc_noise_curve.json"), "w") as f:
            json.dump(bc, f, indent=2)
        print(f"Saved results/bc_noise_curve.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
