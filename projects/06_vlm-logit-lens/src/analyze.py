"""Read logit_lens_results.csv and produce 7 figures + tables + summary.

CSV is auto-located: $LOGIT_LENS_CSV, ./logit_lens_results.csv,
./results/logit_lens_results.csv, then a couple of script-relative paths.
Outputs land in the cwd.
"""
import os, sys, time, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_csv():
    here = Path(__file__).parent
    candidates = [
        os.environ.get("LOGIT_LENS_CSV"),
        Path.cwd() / "logit_lens_results.csv",
        Path.cwd() / "results" / "logit_lens_results.csv",
        here.parent / "logit_lens_results.csv",
        here.parent / "results" / "logit_lens_results.csv",
    ]
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        "logit_lens_results.csv not found. Tried: "
        + ", ".join(str(c) for c in candidates if c is not None)
    )


CSV = _find_csv()
OUT = Path.cwd()
TABLES_TXT = OUT / "analysis_tables.txt"
SUMMARY_JSON = OUT / "analysis_summary.json"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "font.size": 11,
})
CATEGORY_COLORS = {
    "color":   "#2196F3",
    "shape":   "#4CAF50",
    "count":   "#FF9800",
    "spatial": "#9C27B0",
    "binding": "#F44336",
}
COLOR_RGB = {"red": "#E53935", "blue": "#1E88E5", "green": "#43A047", "yellow": "#FDD835"}
SHAPE_COLORS = {"circle": "#1f77b4", "square": "#ff7f0e",
                "triangle": "#2ca02c", "star": "#d62728"}
COUNT_COLORS = {"1": "#440154", "2": "#3b528b", "3": "#21908d",
                "4": "#5dc863", "5": "#fde725"}
SPATIAL_COLORS = {"left": "#1f77b4", "right": "#2ca02c",
                  "above": "#9467bd", "below": "#d62728"}
BIND_COLORS = {"red": "#E53935", "blue": "#1E88E5"}

N_LAYERS = 29
EMERGENCE_THRESHOLD = 0.1
N_BOOT = 1000
RNG = np.random.default_rng(42)

# tee printed lines so we can also write them to analysis_tables.txt
_lines = []
def emit(s=""):
    print(s)
    _lines.append(s)


def bootstrap_means(values, n_boot=N_BOOT, rng=None):
    """Mean + 95% CI from a vanilla bootstrap."""
    rng = rng or RNG
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    if n == 1:
        v = float(values[0])
        return (v, v, v)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def per_layer_curve(df, value_col="p_target", n_boot=N_BOOT):
    """For each layer 0..28, bootstrap mean+CI of `value_col`."""
    layers = np.arange(N_LAYERS)
    means = np.zeros(N_LAYERS)
    los = np.zeros(N_LAYERS)
    his = np.zeros(N_LAYERS)
    g = df.groupby("layer")[value_col]
    for L in layers:
        try:
            vals = g.get_group(L).values.astype(float)
        except KeyError:
            means[L] = los[L] = his[L] = np.nan
            continue
        m, lo, hi = bootstrap_means(vals, n_boot=n_boot)
        means[L] = m; los[L] = lo; his[L] = hi
    return layers, means, los, his


def load_data():
    print(f"loading {CSV}...")
    df = pd.read_csv(CSV)
    print(f"  shape={df.shape}  unique_images={df.image_file.nunique()}")
    df["layer"] = df["layer"].astype(int)
    df["rank_target"] = df["rank_target"].astype(int)
    for col in ["p_target", "p_distractor", "margin", "entropy"]:
        df[col] = df[col].astype(float)
    return df


def part_A(df):
    emit("=" * 78)
    emit("PART A — DATA EXPLORATION")
    emit("=" * 78)

    emit("\n[A1] Per-category, per-class accuracy at L28 (rank_target == 0)")
    l28 = df[df.layer == 28]
    a1_records = []
    for cat, sub in l28.groupby("category"):
        emit(f"\n  category={cat}  (n={sub.image_file.nunique()})")
        emit(f"    {'gt_label':<10} {'n':>5} {'acc':>8}  {'mean_P(target)':>14}  {'median rank':>12}")
        for gt, ssub in sub.groupby("gt_label"):
            n = len(ssub)
            acc = (ssub.rank_target == 0).mean()
            ptm = ssub.p_target.mean()
            med_rank = int(ssub.rank_target.median())
            emit(f"    {gt:<10} {n:>5} {acc:>8.1%}  {ptm:>14.4f}  {med_rank:>12d}")
            a1_records.append({"category": cat, "gt_label": gt, "n": n,
                               "acc_L28": float(acc), "mean_p_L28": float(ptm),
                               "median_rank_L28": med_rank})

    emit("\n[A2] Emergence layer  (first layer with mean P(target) > "
         f"{EMERGENCE_THRESHOLD})")
    emit(f"  {'category':<10} {'gt_label':<10} {'emerge_L':>8} {'P@emerge':>9} {'P@L28':>8}")
    a2_records = []
    for (cat, gt), sub in df.groupby(["category", "gt_label"]):
        means_by_layer = sub.groupby("layer")["p_target"].mean()
        over = means_by_layer[means_by_layer > EMERGENCE_THRESHOLD]
        if len(over) == 0:
            emL = "—"; pemerge = float("nan")
            emerge_str = "  —  "
        else:
            emL = int(over.index.min())
            pemerge = float(means_by_layer.loc[emL])
            emerge_str = f"{emL:>8}"
        p28 = float(means_by_layer.loc[28]) if 28 in means_by_layer.index else float("nan")
        emit(f"  {cat:<10} {gt:<10} {emerge_str} "
             f"{(f'{pemerge:.3f}' if not np.isnan(pemerge) else '   nan'):>9} "
             f"{p28:>8.3f}")
        a2_records.append({"category": cat, "gt_label": gt,
                           "emergence_layer": (None if isinstance(emL, str) else int(emL)),
                           "P_at_emergence": (None if np.isnan(pemerge) else pemerge),
                           "P_at_L28": p28})

    emit("\n[A3] Binding analysis (per layer; mean across pairs)")
    emit(f"  {'layer':>5} {'P_A(red)':>10} {'P_B(blue)':>10} "
         f"{'distrA':>10} {'distrB':>10} {'margin':>10} {'pair_acc':>10}")
    bind = df[df.category == "binding"].copy()
    bind_a = bind[bind.pair_side == "A"]
    bind_b = bind[bind.pair_side == "B"]
    pairs = bind.pair_id.unique()
    a3_records = []
    pair_acc_by_layer = {}
    for L in range(N_LAYERS):
        sa = bind_a[bind_a.layer == L]
        sb = bind_b[bind_b.layer == L]
        # pair-level acc requires BOTH A and B correct on the same pair
        merged = sa.merge(sb, on="pair_id", suffixes=("_A", "_B"))
        pair_acc = ((merged.rank_target_A == 0) & (merged.rank_target_B == 0)).mean()
        rec = {
            "layer": L,
            "p_A_red":   float(sa.p_target.mean()),
            "p_B_blue":  float(sb.p_target.mean()),
            "p_A_distractor": float(sa.p_distractor.mean()),
            "p_B_distractor": float(sb.p_distractor.mean()),
            "margin":    float(bind[bind.layer == L].margin.mean()),
            "pair_acc":  float(pair_acc),
        }
        a3_records.append(rec)
        pair_acc_by_layer[L] = pair_acc
    # show every other layer + the transition window
    for r in a3_records:
        if r["layer"] % 2 == 0 or r["layer"] in (1, 21, 22, 23, 24, 25):
            emit(f"  {r['layer']:>5} {r['p_A_red']:>10.4f} {r['p_B_blue']:>10.4f} "
                 f"{r['p_A_distractor']:>10.4f} {r['p_B_distractor']:>10.4f} "
                 f"{r['margin']:>10.4f} {r['pair_acc']:>10.1%}")

    pair_emerge = next((L for L, a in pair_acc_by_layer.items() if a > 0.5), None)
    pair_acc_28 = pair_acc_by_layer[28]
    emit(f"\n  pair_acc @ L28 = {pair_acc_28:.1%}  (n_pairs={len(pairs)})")
    emit(f"  first layer with pair_acc > 50%: L{pair_emerge}")

    emit("\n[A4] Count per-class mean P(target) by layer")
    cnt = df[df.category == "count"]
    a4_grid = (cnt.groupby(["gt_label", "layer"])["p_target"]
               .mean().unstack("gt_label"))
    a4_grid = a4_grid[["1", "2", "3", "4", "5"]]
    emit("  layer |  one    two    three  four   five")
    for L in range(N_LAYERS):
        row = a4_grid.loc[L]
        emit(f"  {L:>5} | {row['1']:.3f}  {row['2']:.3f}  {row['3']:.3f}  "
             f"{row['4']:.3f}  {row['5']:.3f}")

    emit("\n[A5] Wrong-answer analysis at L28")
    cnt28 = cnt[cnt.layer == 28]
    for gt in ["4", "5"]:
        wrong = cnt28[(cnt28.gt_label == gt) & (cnt28.rank_target > 0)]
        if len(wrong) == 0:
            emit(f"\n  count gt={gt}: 0 wrong cases")
            continue
        c = Counter(wrong.top1_token.tolist())
        emit(f"\n  count gt={gt}, n_wrong={len(wrong)}/{(cnt28.gt_label==gt).sum()}: top-1 distribution")
        for tok, n in c.most_common(8):
            emit(f"    {tok!r:<20}  {n:>4}  ({n/len(wrong):.1%})")

    spat = df[df.category == "spatial"]
    spat28 = spat[spat.layer == 28]
    for gt in ["below", "above", "left", "right"]:
        sub = spat28[spat28.gt_label == gt]
        wrong = sub[sub.rank_target > 0]
        if len(wrong) == 0:
            emit(f"\n  spatial gt={gt}: 0 wrong cases")
            continue
        c = Counter(wrong.top1_token.tolist())
        emit(f"\n  spatial gt={gt}, n_wrong={len(wrong)}/{len(sub)}: top-1 distribution")
        for tok, n in c.most_common(8):
            emit(f"    {tok!r:<20}  {n:>4}  ({n/len(wrong):.1%})")

    return {
        "A1": a1_records,
        "A2": a2_records,
        "A3": a3_records,
        "A3_pair_emerge_layer": pair_emerge,
        "A3_pair_acc_L28": pair_acc_28,
        "A4_grid": a4_grid.to_dict(),
    }


def _plot_lines_with_ci(ax, df, group_col, classes, palette, value_col="p_target",
                        n_boot=N_BOOT):
    for cls in classes:
        sub = df[df[group_col] == cls]
        if len(sub) == 0:
            continue
        layers, means, lo, hi = per_layer_curve(sub, value_col=value_col, n_boot=n_boot)
        color = palette.get(cls, "#444")
        ax.plot(layers, means, color=color, lw=2, label=cls)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.15, linewidth=0)


def fig1_emergence(df, a1_records, a2_records):
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), sharey=True)
    cats = ["color", "shape", "count", "spatial", "binding"]
    palettes = {
        "color":   COLOR_RGB,
        "shape":   SHAPE_COLORS,
        "count":   COUNT_COLORS,
        "spatial": SPATIAL_COLORS,
        "binding": BIND_COLORS,
    }
    class_order = {
        "color":   ["red", "blue", "green", "yellow"],
        "shape":   ["circle", "square", "triangle", "star"],
        "count":   ["1", "2", "3", "4", "5"],
        "spatial": ["left", "right", "above", "below"],
        "binding": ["red", "blue"],
    }
    a1_by_cat = defaultdict(list)
    for r in a1_records:
        a1_by_cat[r["category"]].append(r)
    for ax, cat in zip(axes, cats):
        sub = df[df.category == cat]
        _plot_lines_with_ci(ax, sub, "gt_label", class_order[cat], palettes[cat])
        ems = [r["emergence_layer"] for r in a2_records
               if r["category"] == cat and r["emergence_layer"] is not None]
        if ems:
            mean_em = int(round(np.mean(ems)))
            ax.axvline(mean_em, color="gray", lw=1, ls="--", alpha=0.7)
            ax.text(mean_em + 0.3, 0.95, f"mean emerge L{mean_em}",
                    color="gray", fontsize=9, transform=ax.get_xaxis_transform())
        l28_sub = sub[sub.layer == 28]
        acc = (l28_sub.rank_target == 0).mean() if len(l28_sub) else 0.0
        ax.set_title(f"{cat}  (L28 acc={acc:.0%})", fontsize=12)
        ax.set_xlabel("layer")
        ax.set_xlim(0, 28)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    axes[0].set_ylabel("mean P(target)")
    fig.suptitle("Logit-lens emergence curves — Qwen2-VL-2B-Instruct", fontsize=14, y=1.02)
    fig.tight_layout()
    out = OUT / "fig1_emergence_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2_heatmap(df):
    cats = ["color", "shape", "count", "spatial", "binding"]
    class_order = {
        "color":   ["red", "blue", "green", "yellow"],
        "shape":   ["circle", "square", "triangle", "star"],
        "count":   ["1", "2", "3", "4", "5"],
        "spatial": ["left", "right", "above", "below"],
        "binding": ["red", "blue"],
    }
    columns = []
    cat_boundaries = []
    for cat in cats:
        for cls in class_order[cat]:
            columns.append((cat, cls))
        cat_boundaries.append(len(columns))
    mat = np.full((N_LAYERS, len(columns)), np.nan)
    for i, (cat, cls) in enumerate(columns):
        sub = df[(df.category == cat) & (df.gt_label == cls)]
        means = sub.groupby("layer")["p_target"].mean()
        for L in range(N_LAYERS):
            if L in means.index:
                mat[L, i] = means.loc[L]
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{L}" for L in range(N_LAYERS)])
    ax.set_xticks(range(len(columns)))
    labels = [f"{cls}\n[{cat}]" for cat, cls in columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for b in cat_boundaries[:-1]:
        ax.axvline(b - 0.5, color="white", lw=2)
    for L in range(N_LAYERS):
        for j in range(len(columns)):
            v = mat[L, j]
            if not np.isnan(v) and v > 0.5:
                ax.text(j, L, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.7 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean P(target)")
    ax.set_title("Layer × class heatmap of mean P(target)", fontsize=13)
    fig.tight_layout()
    out = OUT / "fig2_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_binding(df, a3_records):
    bind = df[df.category == "binding"]
    bind_a = bind[bind.pair_side == "A"]   # target = red
    bind_b = bind[bind.pair_side == "B"]   # target = blue

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) P(correct color)
    ax = axes[0, 0]
    layers, mA, loA, hiA = per_layer_curve(bind_a, "p_target")
    _,      mB, loB, hiB = per_layer_curve(bind_b, "p_target")
    ax.plot(layers, mA, color=BIND_COLORS["red"], lw=2, label="A → red")
    ax.fill_between(layers, loA, hiA, color=BIND_COLORS["red"], alpha=0.15, linewidth=0)
    ax.plot(layers, mB, color=BIND_COLORS["blue"], lw=2, label="B → blue")
    ax.fill_between(layers, loB, hiB, color=BIND_COLORS["blue"], alpha=0.15, linewidth=0)
    ax.set_title("(a) P(correct color)")
    ax.set_xlabel("layer"); ax.set_ylabel("P(target)")
    ax.set_xlim(0, 28); ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # (b) P(distractor)
    ax = axes[0, 1]
    _, mAd, loAd, hiAd = per_layer_curve(bind_a, "p_distractor")
    _, mBd, loBd, hiBd = per_layer_curve(bind_b, "p_distractor")
    ax.plot(layers, mAd, color=BIND_COLORS["blue"], lw=2, label="A → P(blue) [distractor]")
    ax.fill_between(layers, loAd, hiAd, color=BIND_COLORS["blue"], alpha=0.15, linewidth=0)
    ax.plot(layers, mBd, color=BIND_COLORS["red"], lw=2, label="B → P(red) [distractor]")
    ax.fill_between(layers, loBd, hiBd, color=BIND_COLORS["red"], alpha=0.15, linewidth=0)
    ax.set_title("(b) P(distractor color)")
    ax.set_xlabel("layer"); ax.set_ylabel("P(distractor)")
    ax.set_xlim(0, 28); ax.set_ylim(-0.02, 1.02)
    ax.legend()

    # (c) margin
    ax = axes[1, 0]
    _, mm, lom, him = per_layer_curve(bind, "margin")
    ax.plot(layers, mm, color="#444", lw=2, label="P(target) − P(distractor_max)")
    ax.fill_between(layers, lom, him, color="#444", alpha=0.15, linewidth=0)
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_title("(c) Margin (target minus best distractor)")
    ax.set_xlabel("layer"); ax.set_ylabel("margin")
    ax.set_xlim(0, 28)
    ax.legend()

    # (d) pair-level accuracy
    ax = axes[1, 1]
    pair_acc = np.array([r["pair_acc"] for r in a3_records])
    ax.plot(layers, pair_acc, color="#7B1FA2", lw=2, label="both A and B correct")
    ax.fill_between(layers, np.zeros_like(pair_acc), pair_acc, color="#7B1FA2", alpha=0.1, linewidth=0)
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
    pair_emerge = next((L for L in range(N_LAYERS) if pair_acc[L] > 0.5), None)
    if pair_emerge is not None:
        ax.axvline(pair_emerge, color="gray", lw=1, ls="--", alpha=0.7)
        ax.text(pair_emerge + 0.3, 0.55, f"L{pair_emerge}", color="gray")
    ax.set_title(f"(d) Pair-level accuracy  (L28 = {pair_acc[28]:.1%})")
    ax.set_xlabel("layer"); ax.set_ylabel("pair accuracy")
    ax.set_xlim(0, 28); ax.set_ylim(-0.02, 1.02)
    ax.legend()

    fig.suptitle("Binding analysis — Qwen2-VL-2B-Instruct", fontsize=14, y=1.00)
    fig.tight_layout()
    out = OUT / "fig3_binding.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_entropy(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["color", "shape", "count", "spatial", "binding"]
    annotations = {}
    for cat in cats:
        sub = df[df.category == cat]
        layers, m, lo, hi = per_layer_curve(sub, "entropy")
        ax.plot(layers, m, color=CATEGORY_COLORS[cat], lw=2, label=cat)
        ax.fill_between(layers, lo, hi, color=CATEGORY_COLORS[cat], alpha=0.15, linewidth=0)
        # layer of the largest single-step entropy drop (argmin returns the END layer)
        diffs = np.diff(m)
        argmin = int(np.argmin(diffs))
        annotations[cat] = (argmin + 1, m[argmin + 1])
    for cat, (L, y) in annotations.items():
        ax.scatter([L], [y], color=CATEGORY_COLORS[cat], zorder=5, s=40,
                   edgecolor="black", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean entropy (nats)")
    ax.set_title("Entropy dynamics by category (markers = layer of largest single-step drop)")
    ax.set_xlim(0, 28)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = OUT / "fig4_entropy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, annotations


def fig5_count(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    cnt = df[df.category == "count"]
    NAME = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
    for gt in ["1", "2", "3", "4", "5"]:
        sub = cnt[cnt.gt_label == gt]
        layers, m, lo, hi = per_layer_curve(sub, "p_target")
        ax.plot(layers, m, color=COUNT_COLORS[gt], lw=2, label=NAME[gt])
        ax.fill_between(layers, lo, hi, color=COUNT_COLORS[gt], alpha=0.15, linewidth=0)
    ax.set_xlabel("layer"); ax.set_ylabel("mean P(target)")
    ax.set_title("Count: per-class mean P(target) trajectory")
    ax.set_xlim(0, 28); ax.set_ylim(-0.02, 1.02)
    ax.legend(title="ground truth")
    fig.tight_layout()
    out = OUT / "fig5_count_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6_spatial(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    spat = df[df.category == "spatial"]
    for rel in ["left", "right", "above", "below"]:
        sub = spat[spat.gt_label == rel]
        layers, m, lo, hi = per_layer_curve(sub, "p_target")
        ax.plot(layers, m, color=SPATIAL_COLORS[rel], lw=2, label=rel)
        ax.fill_between(layers, lo, hi, color=SPATIAL_COLORS[rel], alpha=0.15, linewidth=0)
    ax.set_xlabel("layer"); ax.set_ylabel("mean P(target)")
    ax.set_title("Spatial: per-class mean P(target) trajectory")
    ax.set_xlim(0, 28); ax.set_ylim(-0.02, 1.02)
    ax.legend(title="relation")
    fig.tight_layout()
    out = OUT / "fig6_spatial_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_rank(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["color", "shape", "count", "spatial", "binding"]
    for cat in cats:
        sub = df[df.category == cat]
        layers = np.arange(N_LAYERS)
        means = np.zeros(N_LAYERS)
        for L in layers:
            means[L] = sub[sub.layer == L].rank_target.mean()
        ax.plot(layers, means + 1, color=CATEGORY_COLORS[cat], lw=2, label=cat)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("layer"); ax.set_ylabel("mean rank(target) + 1  (log, inverted)")
    ax.set_title("Rank trajectory of target token")
    ax.set_xlim(0, 28)
    ax.legend()
    fig.tight_layout()
    out = OUT / "fig7_rank_trajectories.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    t0 = time.time()
    df = load_data()

    summary = {}
    summary["partA"] = part_A(df)

    print("\n[plots] generating figures...")
    p1 = fig1_emergence(df, summary["partA"]["A1"], summary["partA"]["A2"])
    p2 = fig2_heatmap(df)
    p3 = fig3_binding(df, summary["partA"]["A3"])
    p4, ent_ann = fig4_entropy(df)
    p5 = fig5_count(df)
    p6 = fig6_spatial(df)
    p7 = fig7_rank(df)
    summary["plots"] = {
        "fig1": str(p1), "fig2": str(p2), "fig3": str(p3),
        "fig4": str(p4), "fig5": str(p5), "fig6": str(p6), "fig7": str(p7),
    }
    summary["entropy_drop_layer_per_category"] = {k: int(v[0]) for k, v in ent_ann.items()}

    TABLES_TXT.write_text("\n".join(_lines))
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    emit("")
    emit(f"[done] total {elapsed:.1f}s")
    emit(f"[done] tables  -> {TABLES_TXT}")
    emit(f"[done] summary -> {SUMMARY_JSON}")
    for p in summary["plots"].values():
        emit(f"[done] plot    -> {p}")

    # final rewrite to capture the [done] lines that emit() appended above
    TABLES_TXT.write_text("\n".join(_lines))


if __name__ == "__main__":
    main()
