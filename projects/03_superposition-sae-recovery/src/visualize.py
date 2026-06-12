import numpy as np
import torch
import matplotlib.pyplot as plt


FIGSIZE = (8, 6)
CMAP = "viridis"


def plot_loss_curve(losses, title="Training Loss", save_path=None, log_scale=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(losses, linewidth=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_W_columns_2d(W, p, save_path=None):
    """Plot feature embeddings on unit circle (d=2 only). Size ~ ||W_i||."""
    W_np = W.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()

    norms = np.linalg.norm(W_np, axis=1, keepdims=True)
    W_unit = W_np / norms

    fig, ax = plt.subplots(figsize=(8, 8))

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.3)

    norms_flat = norms.squeeze()
    size_min, size_max = 30, 300
    if norms_flat.max() > norms_flat.min():
        sizes = size_min + (size_max - size_min) * (
            (norms_flat - norms_flat.min()) / (norms_flat.max() - norms_flat.min())
        )
    else:
        sizes = np.full_like(norms_flat, (size_min + size_max) / 2)

    log_p = np.log10(p_np)
    sc = ax.scatter(W_unit[:, 0], W_unit[:, 1], c=log_p, cmap=CMAP,
                    s=sizes, zorder=3, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="log10(p)")

    for frac, label in [(0.25, "small"), (0.5, "med"), (1.0, "large")]:
        val = norms_flat.min() + frac * (norms_flat.max() - norms_flat.min())
        s = size_min + (size_max - size_min) * frac
        ax.scatter([], [], c="gray", s=s, edgecolors="k", linewidths=0.5,
                   label=f"||W||={val:.2f}")
    ax.legend(loc="upper left", title="Norm scale", fontsize=8,
              title_fontsize=9, framealpha=0.8)

    top5 = np.argsort(-p_np)[:5]
    for i in top5:
        ax.annotate(str(i), (W_unit[i, 0], W_unit[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_aspect("equal")
    ax.set_title("Feature Embeddings on Unit Circle")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cosine_similarity_hist(cos_sim_matrix, title="Off-diagonal Cosine Similarities",
                                save_path=None):
    mat = cos_sim_matrix.detach().cpu().numpy()
    F = mat.shape[0]
    mask = ~np.eye(F, dtype=bool)
    off_diag = mat[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(off_diag, bins=50, edgecolor="k", linewidth=0.3, alpha=0.7)

    for val, label in [(0, "0"), (0.5, "0.5"), (-0.5, "-0.5"),
                       (1.0, "1"), (-1.0, "-1")]:
        ax.axvline(val, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(val, ax.get_ylim()[1] * 0.95, label, ha="center",
                fontsize=8, color="r")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_dimensionality(D, p, save_path=None):
    order = torch.argsort(-p)
    D_sorted = D[order].detach().cpu().numpy()
    p_sorted = p[order].detach().cpu().numpy()
    log_p = np.log10(p_sorted)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm = plt.Normalize(log_p.min(), log_p.max())
    colors = plt.cm.get_cmap(CMAP)(norm(log_p))
    ax.bar(range(len(D_sorted)), D_sorted, color=colors, edgecolor="k",
           linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    plt.colorbar(sm, ax=ax, label="log10(p)")

    for val, label in [(1.0, "dedicated"), (0.5, "digon"), (2 / 3, "triangle")]:
        ax.axhline(val, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(len(D_sorted) - 0.5, val + 0.02, label, ha="right",
                fontsize=8, color="r")

    ax.set_xlabel("Feature Index (sorted by frequency)")
    ax.set_ylabel("Dimensionality")
    ax.set_title("Feature Dimensionality")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_W_norms(W, p, save_path=None):
    norms = W.detach().cpu().norm(dim=1).numpy()
    log_p = np.log10(p.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(log_p, norms, s=40, edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xlabel("log10(p)")
    ax.set_ylabel("||W_i||")
    ax.set_title("Feature Norm vs Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ev_vs_param(param_values, ev_values, param_name, save_path=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ev = np.array(ev_values)

    if ev.ndim == 2:
        mean = ev.mean(axis=0)
        std = ev.std(axis=0)
        ax.errorbar(param_values, mean, yerr=std, marker="o", capsize=4,
                    linewidth=1.5, markersize=6)
    else:
        ax.plot(param_values, ev, marker="o", linewidth=1.5, markersize=6)

    ax.set_xlabel(param_name)
    ax.set_ylabel("Explained Variance")
    ax.set_title(f"Explained Variance vs {param_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frequency_recovery(true_freqs, sae_freqs, save_path=None):
    tf = np.log10(true_freqs.detach().cpu().numpy().clip(1e-10))
    sf = np.log10(sae_freqs.detach().cpu().numpy().clip(1e-10))
    indices = np.arange(len(tf))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(tf, sf, c=indices, cmap=CMAP, s=40, edgecolors="k",
                    linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Feature Index")

    lo = min(tf.min(), sf.min())
    hi = max(tf.max(), sf.max())
    margin = 0.1 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("log10(true freq)")
    ax.set_ylabel("log10(SAE freq)")
    ax.set_title("Frequency Recovery")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_mmcs_per_feature(W_gt, W_dec, assignments, p, save_path=None):
    W_gt_n = W_gt / W_gt.norm(dim=1, keepdim=True)
    matched = W_dec[:, assignments].T  # (F, d)
    matched_n = matched / matched.norm(dim=1, keepdim=True)
    cos_sims = (W_gt_n * matched_n).sum(dim=1).abs().detach().cpu().numpy()

    log_p = np.log10(p.detach().cpu().numpy())
    F = len(cos_sims)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm = plt.Normalize(log_p.min(), log_p.max())
    colors = plt.cm.get_cmap(CMAP)(norm(log_p))
    ax.bar(range(F), cos_sims, color=colors, edgecolor="k", linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    plt.colorbar(sm, ax=ax, label="log10(p)")

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("MMCS per Feature")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_weight_cosine_similarity(W, title="W^T W (cosine similarity)", save_path=None):
    from metrics import cosine_similarity_matrix
    mat = cosine_similarity_matrix(W).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_geometry(W, dims=(0, 1), title="Feature Directions (2D projection)",
                          save_path=None):
    W_np = W.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(W_np[:, dims[0]], W_np[:, dims[1]], s=40, edgecolors="k",
               linewidths=0.5, zorder=3)
    for i in range(len(W_np)):
        ax.annotate(str(i), (W_np[i, dims[0]], W_np[i, dims[1]]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f"Dim {dims[0]}")
    ax.set_ylabel(f"Dim {dims[1]}")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_superposition_metric_vs_alpha(alphas, metric_values, metric_name="MMCS",
                                       title=None, save_path=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(alphas, metric_values, marker="o", linewidth=1.5, markersize=6)
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} vs alpha")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sae_dictionary_elements(W_dec, W_true, top_k=10,
                                  title="SAE Dictionary vs True Features",
                                  save_path=None):
    W_dec_n = W_dec / W_dec.norm(dim=1, keepdim=True)
    W_true_n = W_true / W_true.norm(dim=1, keepdim=True)
    cos_sim = (W_dec_n @ W_true_n.T).abs()
    max_cos, matched = cos_sim.max(dim=1)
    top_idx = torch.argsort(-max_cos)[:top_k]

    fig, axes = plt.subplots(2, top_k, figsize=(2 * top_k, 4))
    for col, idx in enumerate(top_idx):
        for row, (data, label) in enumerate([
            (W_dec[idx].detach().cpu(), "SAE"),
            (W_true[matched[idx]].detach().cpu(), "GT"),
        ]):
            ax = axes[row, col]
            ax.bar(range(len(data)), data.numpy(), width=0.8)
            ax.set_title(f"{label} (cos={max_cos[idx]:.2f})", fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dead_latents_vs_l1(l1_coeffs, dead_fractions,
                             title="Dead Latents vs L1 Coefficient", save_path=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(l1_coeffs, dead_fractions, marker="o", linewidth=1.5, markersize=6)
    ax.set_xlabel("L1 Coefficient")
    ax.set_ylabel("Fraction Dead Latents")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sweep_heatmap(row_values, col_values, metric_grid, row_label, col_label,
                       metric_label, title="Parameter Sweep", save_path=None):
    grid = metric_grid.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=CMAP)
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([f"{v:.3g}" for v in col_values], rotation=45)
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([f"{v:.3g}" for v in row_values])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=metric_label)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
