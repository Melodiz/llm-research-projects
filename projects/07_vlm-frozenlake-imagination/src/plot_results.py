from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/A7_final_assembly")
A5_DIR = Path("results/A5_world_model_eval")
A6_DIR = Path("results/A6_planning_comparison")


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def rate(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_part2_tables() -> None:
    rows = []
    for condition in ("image_text", "image_only"):
        metrics = load_json(A5_DIR / condition / "metrics.json")
        label = "Image + GT state text" if condition == "image_text" else "Image-only"
        for split in ("train", "val"):
            m = metrics[split]
            rows.append(
                {
                    "condition": label,
                    "split": split,
                    "count": m["count"],
                    "format_compliance": rate(m["format_compliance_rate"]),
                    "exact_match": rate(m["exact_target_match_rate"]),
                    "position_acc": rate(m["position_accuracy"]),
                    "outcome_acc": rate(m["outcome_accuracy"]),
                    "both_correct": rate(m["both_correct_accuracy"]),
                    "wrong_position": m["wrong_position_count"],
                    "wrong_outcome": m["wrong_outcome_count"],
                    "wrong_both": m["wrong_both_count"],
                    "format_errors": m["format_error_count"],
                    "calibration": metrics["train_calibration_verdict"],
                }
            )
    write_csv(OUT_DIR / "part2_metrics_table.csv", rows)
    write_markdown_table(OUT_DIR / "part2_metrics_table.md", rows)

    for condition in ("image_text", "image_only"):
        src = A5_DIR / condition / "confusion_matrix_val.csv"
        dst = OUT_DIR / f"part2_confusion_{condition}_val.csv"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def make_part3_tables_and_plot() -> None:
    ci = load_json(A6_DIR / "bootstrap_ci.json")
    rows = []
    with (A6_DIR / "comparison_table.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            rows.append(
                {
                    "condition": condition,
                    "episodes": int(row["episodes"]),
                    "success_rate": rate(float(row["success_rate"])),
                    "success_95ci": f"[{ci[condition]['success_rate']['low']:.3f}, {ci[condition]['success_rate']['high']:.3f}]",
                    "mean_steps": f"{float(row['mean_steps']):.2f}",
                    "mean_steps_95ci": f"[{ci[condition]['mean_steps']['low']:.2f}, {ci[condition]['mean_steps']['high']:.2f}]",
                    "hole_rate": rate(float(row["hole_rate"])),
                    "hole_95ci": f"[{ci[condition]['hole_rate']['low']:.3f}, {ci[condition]['hole_rate']['high']:.3f}]",
                    "truncation_rate": rate(float(row["truncation_rate"])),
                    "format_compliance": rate(float(row["format_compliance_rate"])),
                    "parse_failure": rate(float(row["prediction_parse_failure_rate"])),
                    "fallback_rate": rate(float(row["fallback_action_rate"])),
                    "runtime_s": f"{float(row['runtime_seconds']):.1f}",
                }
            )
    write_csv(OUT_DIR / "part3_comparison_table.csv", rows)
    write_markdown_table(OUT_DIR / "part3_comparison_table.md", rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment.
        (OUT_DIR / "part3_success_rate_ci_plot_skipped.txt").write_text(
            f"matplotlib unavailable: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return

    labels = [row["condition"] for row in rows]
    values = [float(row["success_rate"]) for row in rows]
    lows = [ci[label]["success_rate"]["low"] for label in labels]
    highs = [ci[label]["success_rate"]["high"] for label in labels]
    lower_err = [v - low for v, low in zip(values, lows, strict=True)]
    upper_err = [high - v for v, high in zip(values, highs, strict=True)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(labels)), values, color=["#6b7280", "#9ca3af", "#2563eb", "#0f766e", "#111827"])
    ax.errorbar(
        range(len(labels)),
        values,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1,
    )
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Success rate")
    ax.set_title("Part 3 success rate with bootstrap 95% CIs")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part3_success_rate_ci.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_part2_tables()
    make_part3_tables_and_plot()
    summary = {
        "part2_source": str(A5_DIR),
        "part3_source": str(A6_DIR),
        "outputs": sorted(path.name for path in OUT_DIR.iterdir()),
    }
    (OUT_DIR / "source_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote A7 derived artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
