"""Sweep runner: trains toy model + SAE across parameter configurations."""

import os
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import ToyModelConfig, SAEConfig
from toy_model import train_toy_model, compute_feature_probabilities, generate_batch
from sae import train_sae
from metrics import (
    explained_variance, mean_max_cosine_similarity, weighted_mmcs,
    fraction_dead_latents, cosine_similarity_matrix, feature_dimensionality,
    feature_recovery_rate, frequency_recovery, l0_sparsity,
)
from visualize import (
    plot_loss_curve, plot_W_columns_2d, plot_cosine_similarity_hist,
    plot_feature_dimensionality, plot_W_norms, plot_ev_vs_param,
    plot_frequency_recovery, plot_mmcs_per_feature,
)


def run_single(toy_cfg, sae_cfg, save_dir, device):
    """Train toy model + SAE, compute all metrics, save plots. Returns results dict."""
    os.makedirs(save_dir, exist_ok=True)
    p = compute_feature_probabilities(toy_cfg.F, toy_cfg.alpha)

    model, tm_losses = train_toy_model(toy_cfg, device)
    plot_loss_curve(tm_losses, "Toy Model Loss", os.path.join(save_dir, "tm_loss.png"))
    plt.close("all")

    W = model.W.data
    cos_mat = cosine_similarity_matrix(W)
    D = feature_dimensionality(W)

    plot_cosine_similarity_hist(cos_mat, "Cos Sim", os.path.join(save_dir, "cos_hist.png"))
    plot_feature_dimensionality(D, p, os.path.join(save_dir, "feat_dim.png"))
    plot_W_norms(W, p, os.path.join(save_dir, "w_norms.png"))
    if toy_cfg.d == 2:
        plot_W_columns_2d(W, p, os.path.join(save_dir, "w_2d.png"))
    plt.close("all")

    sae, sae_losses = train_sae(model, toy_cfg, sae_cfg, device)
    plot_loss_curve(sae_losses, "SAE Loss", os.path.join(save_dir, "sae_loss.png"))
    plt.close("all")

    torch.manual_seed(0)
    x = generate_batch(toy_cfg, 10000, device)
    with torch.no_grad():
        h = model.encode(x)
        h_hat, z = sae(h)

    ev = explained_variance(h, h_hat)
    mmcs_val, assignments = mean_max_cosine_similarity(W, sae.W_dec.data.T)
    wmmcs_val, _ = weighted_mmcs(W, sae.W_dec.data.T)
    dead = fraction_dead_latents(z)
    frr = feature_recovery_rate(W, sae.W_dec.data.T)
    l0 = l0_sparsity(z)

    true_f, sae_f = frequency_recovery(
        p, W, sae.W_dec.data, sae.W_enc.data, sae.b_enc.data,
        W, toy_cfg.alpha, toy_cfg.F,
    )

    plot_frequency_recovery(true_f, sae_f, os.path.join(save_dir, "freq_recovery.png"))
    plot_mmcs_per_feature(W, sae.W_dec.data, assignments, p,
                          os.path.join(save_dir, "mmcs_per_feat.png"))
    plt.close("all")

    return {
        "F": toy_cfg.F, "d": toy_cfg.d, "alpha": toy_cfg.alpha,
        "F_sae": sae_cfg.F_sae, "l0_coeff": sae_cfg.l0_coeff, "seed": toy_cfg.seed,
        "EV": ev, "MMCS": mmcs_val, "weighted_MMCS": wmmcs_val,
        "dead_frac": dead, "feature_recovery": frr, "l0": l0,
        "tm_final_loss": tm_losses[-1], "sae_final_loss": sae_losses[-1],
    }


def sweep(param_name, param_values, base_toy_cfg, base_sae_cfg, device,
          seeds=[42, 43, 44], save_dir="results", sweep_label=None):
    """Sweep one parameter across multiple seeds. Returns DataFrame."""
    sweep_name = sweep_label or f"sweep_{param_name}"
    sweep_dir = os.path.join(save_dir, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    rows = []
    for val in param_values:
        for seed in seeds:
            toy_kwargs, sae_kwargs = {}, {}

            if param_name in ("F", "d", "alpha"):
                toy_kwargs[param_name] = val
            elif param_name == "l0_coeff":
                sae_kwargs["l0_coeff"] = val
            elif param_name == "F_sae":
                sae_kwargs["F_sae"] = val

            toy_cfg = replace(base_toy_cfg, seed=seed, **toy_kwargs)
            sae_cfg = replace(base_sae_cfg, seed=seed, **sae_kwargs)

            run_dir = os.path.join(sweep_dir, f"{param_name}={val}_seed={seed}")
            result = run_single(toy_cfg, sae_cfg, run_dir, device)
            result["param_name"] = param_name
            result["param_value"] = val
            rows.append(result)

            print(f"[{sweep_name}] {param_name}={val}, seed={seed} "
                  f"— EV={result['EV']:.4f}, MMCS={result['MMCS']:.4f}, "
                  f"wMMCS={result['weighted_MMCS']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sweep_dir, "results.csv"), index=False)
    return df


def _plot_sweep_metric(df, param_name, metric, save_dir):
    grouped = df.groupby("param_value")[metric]
    vals = grouped.mean().index.tolist()
    means = grouped.mean().values
    stds = grouped.std().values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(vals, means, yerr=stds, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {param_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{metric}_vs_{param_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 43, 44]
    save_root = "results"

    base_toy = ToyModelConfig(F=50, d=5, alpha=1.0, train_steps=2_000)
    base_sae = SAEConfig(F_sae=100, l0_coeff=0.1, train_steps=4_000)

    def sp(label):
        return os.path.join(save_root, label)

    def plot_metrics(df, param, metrics, label):
        for m in metrics:
            _plot_sweep_metric(df, param, m, sp(label))

    STD_METRICS = ["EV", "MMCS", "weighted_MMCS"]

    # Part 1: geometry
    print("\n--- 1a. Alpha sweep ---")
    df_alpha = sweep("alpha", [0.5, 1.0, 1.5, 2.0],
                     base_toy, base_sae, device, seeds, save_root,
                     sweep_label="1a_alpha")
    plot_metrics(df_alpha, "alpha", STD_METRICS, "1a_alpha")

    print("\n--- 1b. F/d sweep ---")
    df_F = sweep("F", [10, 20, 50, 100],
                 replace(base_toy, d=5, alpha=1.0),
                 base_sae, device, seeds, save_root,
                 sweep_label="1b_F")
    plot_metrics(df_F, "F", STD_METRICS, "1b_F")

    print("\n--- 1c. d sweep ---")
    df_d = sweep("d", [2, 5, 10],
                 replace(base_toy, F=50, alpha=1.0),
                 base_sae, device, seeds, save_root,
                 sweep_label="1c_d")
    plot_metrics(df_d, "d", STD_METRICS, "1c_d")

    print("\n--- 1d. d=2 polytope sweep ---")
    polytope_toy = replace(base_toy, d=2, alpha=1.0)
    polytope_sae = replace(base_sae, F_sae=20)
    df_polytope = sweep("F", [4, 5, 6, 8, 10],
                        polytope_toy, polytope_sae, device, seeds, save_root,
                        sweep_label="1d_polytope_F")
    plot_metrics(df_polytope, "F", STD_METRICS, "1d_polytope_F")

    # Part 2: SAE quality
    print("\n--- 2a. l0_coeff sweep ---")
    df_l0 = sweep("l0_coeff", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                  base_toy, base_sae, device, seeds, save_root,
                  sweep_label="2a_l0_coeff")
    plot_metrics(df_l0, "l0_coeff",
                 STD_METRICS + ["dead_frac", "feature_recovery"], "2a_l0_coeff")

    print("\n--- 2b. F_sae sweep ---")
    F_base = base_toy.F
    df_fsae = sweep("F_sae", [F_base, 2 * F_base, 4 * F_base, 8 * F_base],
                    base_toy, base_sae, device, seeds, save_root,
                    sweep_label="2b_F_sae")
    plot_metrics(df_fsae, "F_sae", STD_METRICS, "2b_F_sae")

    print("\n--- 2e. d=10 high-l0 sweep ---")
    d10_toy = replace(base_toy, d=10)
    d10_sae = replace(base_sae, F_sae=100)
    df_d10_l0 = sweep("l0_coeff", [0.3, 0.5, 1.0],
                      d10_toy, d10_sae, device, seeds, save_root,
                      sweep_label="2e_d10_l0")
    plot_metrics(df_d10_l0, "l0_coeff",
                 STD_METRICS + ["dead_frac"], "2e_d10_l0")

    # Part 3: feature recovery (plots from earlier sweeps)
    print("\n--- 3b. Frequency recovery at best l0 ---")
    best_l0 = df_l0.groupby("param_value")["weighted_MMCS"].mean().idxmax()
    print(f"Best l0_coeff by weighted_MMCS: {best_l0}")

    print("\n--- 3c. MMCS vs F ---")
    plot_metrics(df_F, "F", ["feature_recovery"], "1b_F")

    print("\nAll experiments complete. Results in:", save_root)
