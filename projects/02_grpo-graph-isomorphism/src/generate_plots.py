"""Generate training dynamics plots."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

STYLE = dict(figsize=(10, 6), dpi=150)
plt.rcParams.update({"font.size": 12})


def fig1_base_profile():
    """Bar chart: base model accuracy across difficulties 1-10."""
    df = pd.read_csv("results/base/metrics_per_difficulty.csv")
    x = np.arange(len(df))
    w = 0.25

    fig, ax = plt.subplots(**STYLE)
    ax.bar(x - w, df["iso_accuracy"], w, label="Iso Accuracy", color="#4C72B0")
    ax.bar(x, df["non_iso_accuracy"], w, label="Non-Iso Accuracy", color="#DD8452")
    ax.bar(x + w, df["aggregate_accuracy"], w, label="Aggregate Accuracy", color="#55A868")
    ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="Degenerate floor (33.3%)")

    # Annotate zero iso bars and 100% non-iso bars
    for i in range(len(x)):
        ax.text(x[i] - w, 0.02, "0%", ha="center", va="bottom",
                fontsize=8, color="#4C72B0", fontweight="bold")
        ax.text(x[i], 0.97, "100%", ha="center", va="top",
                fontsize=8, color="#DD8452", fontweight="bold")

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Accuracy")
    ax.set_title("Base Model (Qwen2.5-1.5B) Graph Isomorphism Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["difficulty"])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "base_profile.png"))
    plt.close(fig)
    print("Saved base_profile.png")


def fig2_cpr_dynamics():
    """Line plot: class_prediction_ratio over steps for all GI runs."""
    runs = {
        "Run 1 (Anti-Hack)": ("outputs/run1_antihack/collapse_monitor.csv", "#E24A33"),
        "Run H1 (Hints)": ("outputs/run_h1_hints/collapse_monitor.csv", "#348ABD"),
        "Run S1 (SFT→GRPO)": ("outputs/run_s1_sft_grpo/collapse_monitor.csv", "#55A868"),
    }
    fig, ax = plt.subplots(**STYLE)
    for label, (path, color) in runs.items():
        df = pd.read_csv(path)
        ax.plot(df["step"], df["class_prediction_ratio"], label=label, color=color, linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label='Always "NOT ISO" (CPR=1.0)')
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, label="Balanced (CPR=0.5)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Class Prediction Ratio")
    ax.set_title("Class Prediction Ratio During Training")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cpr_dynamics.png"))
    plt.close(fig)
    print("Saved cpr_dynamics.png")


def fig3_ec_reward():
    """Line plot: EC reward mean over steps with rolling average."""
    df = pd.read_csv("outputs/run_ec/train_metrics.csv")
    df = df.dropna(subset=["reward"])

    fig, ax = plt.subplots(**STYLE)
    ax.plot(df["step"], df["reward"], color="#348ABD", linewidth=1.0,
            alpha=0.4, label="Mean Reward (raw)")

    # 20-step rolling average
    rolling = df["reward"].rolling(window=20, min_periods=1).mean()
    ax.plot(df["step"], rolling, color="#348ABD", linewidth=2.5,
            alpha=0.8, label="20-step Rolling Avg")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Edge Counting: Reward Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ec_reward_curve.png"))
    plt.close(fig)
    print("Saved ec_reward_curve.png")


def fig4_ec_loss_divergence():
    """Dual y-axis plot: loss (left, log) and grad_norm (right, log)."""
    df = pd.read_csv("outputs/run_ec/train_metrics.csv")
    df = df.dropna(subset=["loss", "grad_norm"])
    # Drop outliers below 1e-2 that distort the log scale
    df["loss_pos"] = df["loss"].clip(lower=1e-2)
    df["grad_norm_pos"] = df["grad_norm"].clip(lower=1e-2)

    fig, ax1 = plt.subplots(**STYLE)
    color_loss = "#348ABD"
    color_grad = "#E24A33"

    # Raw lines (low alpha)
    ax1.semilogy(df["step"], df["loss_pos"], color=color_loss, linewidth=1.0,
                 alpha=0.3)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss (log scale)", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.semilogy(df["step"], df["grad_norm_pos"], color=color_grad, linewidth=1.0,
                 alpha=0.25)
    ax2.set_ylabel("Gradient Norm (log scale)", color=color_grad)
    ax2.tick_params(axis="y", labelcolor=color_grad)

    # Smoothed lines (20-step rolling average)
    loss_smooth = df["loss_pos"].rolling(window=20, min_periods=1).mean()
    grad_smooth = df["grad_norm_pos"].rolling(window=20, min_periods=1).mean()
    ax1.semilogy(df["step"], loss_smooth, color=color_loss, linewidth=2.5,
                 alpha=0.85, label="Loss (smoothed)")
    ax2.semilogy(df["step"], grad_smooth, color=color_grad, linewidth=2.5,
                 alpha=0.7, label="Grad Norm (smoothed)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Edge Counting: Loss and Gradient Explosion")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ec_loss_divergence.png"))
    plt.close(fig)
    print("Saved ec_loss_divergence.png")


if __name__ == "__main__":
    fig1_base_profile()
    fig2_cpr_dynamics()
    fig3_ec_reward()
    fig4_ec_loss_divergence()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
