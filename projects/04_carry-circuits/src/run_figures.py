# generates all figures for the report
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from config import N_LAYERS, N_HEADS, N_DIGITS, SEED, ID_TO_CHAR, ANSWER_START
from run_experiments import load_model
from data import load_test_sets, encode_example, classify_example
from counterfactual import load_pairs, ANSWER_LOGIT_POS, ANSWER_SEQ_POS
from patching import _pattern_hook

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TOKEN_LABELS = ["a_H", "a_T", "a_O", "+", "b_H", "b_T", "b_O", "=",
                "s_Ov", "s_H", "s_T", "s_O"]
LAYER_LABELS = ["Before L0", "Before L1", "After L1"]
ANSWER_LABELS = ["overflow", "hundreds", "tens", "ones"]
HEAD_LABELS = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})


def _save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")

def _load_json(path):
    with open(path) as f:
        return json.load(f)


# fig 1 - training curves
def figure_1_training():
    print("\nFigure 1: Training curves")
    log = _load_json(os.path.join(RESULTS_DIR, "training_log.json"))

    train_steps, train_losses = [], []
    eval_steps = []
    eval_accs = {"ba": [], "mc1": [], "us9": [], "s3": []}

    for entry in log:
        if "train_loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["train_loss"])
        elif "eval" in entry:
            eval_steps.append(entry["step"])
            for cat in eval_accs:
                eval_accs[cat].append(entry["eval"]["per_category"].get(cat, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_steps, train_losses, color="#2196F3", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training loss (log scale)")
    ax1.set_title("A. Training Loss")

    colors = {"ba": "#4CAF50", "mc1": "#FF9800", "us9": "#F44336", "s3": "#9C27B0"}
    labels = {"ba": "BA (no carry)", "mc1": "MC1 (single carry)",
              "us9": "US9 (cascade)", "s3": "S3 (full cascade)"}
    for cat in ["ba", "mc1", "us9", "s3"]:
        ax2.plot(eval_steps, eval_accs[cat], color=colors[cat],
                 linewidth=2, label=labels[cat])
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("B. Per-Category Accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, "fig1_training")

# fig 2 - residual stream patching
def figure_2_residual():
    print("\nFigure 2: Residual stream patching")
    data = _load_json(os.path.join(RESULTS_DIR, "exp1_residual_stream.json"))

    subtypes = ["type_a_ones", "type_a_tens", "type_a_hundreds"]
    titles = ["Ones carry", "Tens carry", "Hundreds carry"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

    for ax, st, title in zip(axes, subtypes, titles):
        arr = np.array(data[st])
        vmax = max(abs(arr.min()), abs(arr.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(arr, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(LAYER_LABELS, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Residual stream layer")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Logit diff recovery", fontsize=10)

    _save_fig(fig, "fig2_residual")


# fig 3 - head IE heatmap (main result)
def figure_3_head_ie():
    print("\nFigure 3: Head-level IE (main figure)")
    data = _load_json(os.path.join(RESULTS_DIR, "exp3_head_level_ie.json"))

    circuit_labels = set()
    circ_path = os.path.join(VALIDATION_DIR, "circuit_definition.json")
    if os.path.exists(circ_path):
        circ_def = _load_json(circ_path)
        circuit_labels = set(circ_def.get("circuit_flat", []))

    pair_types = ["type_a_ones", "type_a_tens", "type_a_hundreds", "type_b"]
    titles = ["A. Type A: Ones carry", "B. Type A: Tens carry",
              "C. Type A: Hundreds carry", "D. Type B: Cascade"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    n_heads = N_LAYERS * N_HEADS

    for ax, pt, title in zip(axes.flat, pair_types, titles):
        arr = np.array(data[pt])
        vmax = max(abs(arr.min()), abs(arr.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(arr, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ANSWER_LABELS, fontsize=11)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels(HEAD_LABELS, fontsize=11)
        ax.set_xlabel("Answer digit")
        ax.set_ylabel("Head")
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(n_heads):
            for j in range(4):
                val = arr[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        # highlight circuit heads
        if circuit_labels:
            for i, label in enumerate(HEAD_LABELS):
                if label in circuit_labels:
                    rect = Rectangle((-0.5, i - 0.5), 4, 1,
                                     linewidth=2.5, edgecolor="#FF6F00",
                                     facecolor="none", linestyle="-")
                    ax.add_patch(rect)

    _save_fig(fig, "fig3_head_ie")

# fig 4 - attention patterns
def figure_4_attention(model):
    print("\nFigure 4: Attention patterns")

    a, b = 999, 1
    tokens = encode_example(a, b)
    inp = tokens[:-1].unsqueeze(0)

    token_strs = [ID_TO_CHAR.get(t.item(), "?") for t in tokens[:-1]]
    labels = []
    pos_names = ["a_H", "a_T", "a_O", "+", "b_H", "b_T", "b_O", "=",
                 "s_Ov", "s_H", "s_T", "s_O"]
    for i, ts in enumerate(token_strs):
        labels.append(f"{ts}\n({pos_names[i]})")

    _, cache = model.run_with_cache(inp)

    fig, axes = plt.subplots(N_LAYERS, N_HEADS, figsize=(5 * N_HEADS, 5 * N_LAYERS),
                             constrained_layout=True)

    for l in range(N_LAYERS):
        pattern = cache[_pattern_hook(l)][0]
        for h in range(N_HEADS):
            ax = axes[l, h] if N_LAYERS > 1 else axes[h]
            attn = pattern[h].detach().numpy()
            im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_title(f"L{l}H{h}", fontsize=12)
            if h == N_HEADS - 1:
                fig.colorbar(im, ax=ax, shrink=0.7, label="Attention weight")

    _save_fig(fig, "fig4_attention")


# fig 5 - necessity
def figure_5_necessity():
    print("\nFigure 5: Necessity testing")
    nec_path = os.path.join(VALIDATION_DIR, "necessity.json")
    if not os.path.exists(nec_path):
        print("  Skipping -- necessity.json not found. Run run_validation.py first.")
        return

    nec = _load_json(nec_path)

    circuit_labels = set()
    circ_path = os.path.join(VALIDATION_DIR, "circuit_definition.json")
    if os.path.exists(circ_path):
        circ_def = _load_json(circ_path)
        circuit_labels = set(circ_def.get("circuit_flat", []))

    categories = ["ba", "mc1", "us9", "s3"]
    cat_colors = {"ba": "#4CAF50", "mc1": "#FF9800", "us9": "#F44336", "s3": "#9C27B0"}
    cat_display = {"ba": "BA", "mc1": "MC1", "us9": "US9", "s3": "S3"}

    individual = nec.get("individual", {})
    components = list(individual.keys())

    group = nec.get("group", {})
    extra = []
    for gname in ["all_circuit", "all_non_circuit"]:
        if gname in group:
            extra.append(gname)

    all_names = components + extra

    fig, ax = plt.subplots(1, 1, figsize=(max(14, len(all_names) * 1.2), 6))

    x = np.arange(len(all_names))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for ci, cat in enumerate(categories):
        vals = []
        for name in components:
            vals.append(individual[name][cat]["accuracy"])
        for gname in extra:
            vals.append(group[gname][cat]["accuracy"])

        colors = []
        for name in all_names:
            if name in circuit_labels:
                colors.append(cat_colors[cat])
            else:
                colors.append(cat_colors[cat] + "80")

        bars = ax.bar(x + offsets[ci] * width, vals, width,
                      label=cat_display[cat], color=cat_colors[cat], alpha=0.85)

        for bar, name in zip(bars, all_names):
            if name not in circuit_labels and name not in extra:
                bar.set_alpha(0.4)
            elif name == "all_circuit":
                bar.set_edgecolor("red")
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    xlabels = []
    for name in all_names:
        if name == "all_circuit":
            xlabels.append("ALL\ncircuit")
        elif name == "all_non_circuit":
            xlabels.append("ALL\nnon-circuit")
        else:
            xlabels.append(name)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("")
    ax.legend(fontsize=10, ncol=4, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for i, name in enumerate(all_names):
        if name in circuit_labels:
            ax.text(i, -0.06, "*", ha="center", fontsize=14, color="#FF6F00",
                    fontweight="bold", transform=ax.get_xaxis_transform())

    fig.tight_layout()
    _save_fig(fig, "fig5_necessity")

# fig 6 - sufficiency
def figure_6_sufficiency():
    print("\nFigure 6: Sufficiency testing")
    suff_path = os.path.join(VALIDATION_DIR, "sufficiency.json")
    if not os.path.exists(suff_path):
        print("  Skipping -- sufficiency.json not found. Run run_validation.py first.")
        return

    suff = _load_json(suff_path)
    categories = ["ba", "mc1", "us9", "s3"]
    cat_display = {"ba": "BA", "mc1": "MC1", "us9": "US9", "s3": "S3"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(categories))
    width = 0.3

    full_accs = []
    circ_accs = []
    for cat in categories:
        cat_data = suff["categories"][cat]
        full_accs.append(cat_data["full_model_acc"])
        circ_accs.append(cat_data["circuit_only_acc"])

    ax1.bar(x - width / 2, full_accs, width, label="Full model",
            color="#2196F3", alpha=0.85)
    ax1.bar(x + width / 2, circ_accs, width, label="Circuit only",
            color="#FF9800", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([cat_display[c] for c in categories], fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("A. Full Model vs Circuit-Only", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.1)

    for i, cat in enumerate(categories):
        rec = suff["categories"][cat]["recovery_pct"]
        ax1.text(i + width / 2, circ_accs[i] + 0.02, f"{rec:.0f}%",
                 ha="center", fontsize=9, color="#E65100")

    rand_summary = suff.get("random_recoveries_summary", {})
    if rand_summary:
        n_random = len(next(iter(rand_summary.values()))["values"])
        rand_mean_recoveries = []
        for i in range(n_random):
            r = np.mean([rand_summary[cat]["values"][i] for cat in categories])
            rand_mean_recoveries.append(r)

        actual_recovery = np.mean([suff["categories"][cat]["recovery_pct"]
                                   for cat in categories])

        ax2.hist(rand_mean_recoveries, bins=20, color="#90CAF9", edgecolor="#1565C0",
                 alpha=0.7, label="Random circuits")
        ax2.axvline(actual_recovery, color="#F44336", linewidth=2.5,
                    linestyle="--", label=f"Actual circuit ({actual_recovery:.1f}%)")
        ax2.set_xlabel("Mean recovery (%)", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("B. Circuit vs Random Baselines", fontsize=13)
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No random recovery data", ha="center",
                 va="center", transform=ax2.transAxes)

    fig.tight_layout()
    _save_fig(fig, "fig6_sufficiency")


# fig 7 - component level patching
def figure_7_component():
    print("\nFigure 7: Component-level patching")
    data = _load_json(os.path.join(RESULTS_DIR, "exp2_component_level.json"))

    subtypes = ["type_a_ones", "type_a_tens", "type_a_hundreds"]
    titles = ["A. Ones carry", "B. Tens carry", "C. Hundreds carry"]
    comp_labels = ["L0-Attn", "L0-MLP", "L1-Attn", "L1-MLP"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    x = np.arange(len(ANSWER_LABELS))
    width = 0.18

    for ax, st, title in zip(axes, subtypes, titles):
        for i, (label, color) in enumerate(zip(comp_labels, comp_colors)):
            vals = [data[st][label][al] for al in ANSWER_LABELS]
            ax.bar(x + (i - 1.5) * width, vals, width, label=label,
                   color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ANSWER_LABELS, fontsize=11)
        ax.set_ylabel("Recovery", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylim(-0.1, 1.15)

    _save_fig(fig, "fig7_component")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 50)
    print("  GENERATING FIGURES")
    print("=" * 50)

    figure_1_training()
    figure_2_residual()
    figure_3_head_ie()
    figure_7_component()

    # fig 4 needs model
    ckpt = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
    model = load_model(ckpt)
    figure_4_attention(model)

    # figs 5-6 need validation results
    figure_5_necessity()
    figure_6_sufficiency()

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    for f in sorted(os.listdir(FIGURES_DIR)):
        fpath = os.path.join(FIGURES_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
