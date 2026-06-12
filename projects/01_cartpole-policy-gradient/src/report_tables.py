"""Generate formatted tables (markdown + LaTeX) from HW1 results."""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_grouped(results_dir: str) -> dict:
    from cartpole_pg import load_results
    return load_results(results_dir)


def load_bc(results_dir: str) -> dict:
    path = os.path.join(results_dir, "bc_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_histories(grouped: dict, key: str) -> list:
    records = grouped.get(key, [])
    return [r["history"] for r in records]


def get_configs(grouped: dict, key: str) -> list:
    records = grouped.get(key, [])
    return [r["config"] for r in records]



def solve_episode(history: dict, threshold: float = 475.0,
                  window: int = 5) -> int | None:
    """First episode with `window` consecutive evals >= threshold."""
    evals = history.get("eval_rewards", [])
    if len(evals) < window:
        return None
    for i in range(len(evals) - window + 1):
        if all(evals[i + j]["mean"] >= threshold for j in range(window)):
            return evals[i]["episode"]
    return None



def _bold_md(val: str) -> str:
    return f"**{val}**"


def _bold_tex(val: str) -> str:
    return f"\\textbf{{{val}}}"


def _fmt(val, precision: int = 1) -> str:
    if val is None:
        return "\u2014"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def _find_best(rows: list, col_idx: int, direction: str = "max",
               skip_none: bool = True) -> int | None:
    best_i, best_v = None, None
    for i, row in enumerate(rows):
        v = row[col_idx]
        if v is None and skip_none:
            continue
        if best_v is None:
            best_i, best_v = i, v
        elif direction == "max" and v > best_v:
            best_i, best_v = i, v
        elif direction == "min" and v < best_v:
            best_i, best_v = i, v
    return best_i



def build_markdown_table(headers: list, rows: list, col_aligns: list = None,
                         bold_map: dict = None) -> str:
    if bold_map is None:
        bold_map = {}

    str_rows = []
    for i, row in enumerate(rows):
        str_row = []
        for j, cell in enumerate(row):
            s = _fmt(cell) if not isinstance(cell, str) else cell
            if bold_map.get(j) == i:
                s = _bold_md(s)
            str_row.append(s)
        str_rows.append(str_row)

    widths = [len(h) for h in headers]
    for row in str_rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    lines = []
    hdr = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep_chars = []
    for j in range(len(headers)):
        a = col_aligns[j] if col_aligns else "l"
        if a == "r":
            sep_chars.append("-" * (widths[j] - 1) + ":")
        elif a == "c":
            sep_chars.append(":" + "-" * (widths[j] - 2) + ":")
        else:
            sep_chars.append("-" * widths[j])
    sep = "| " + " | ".join(sep_chars) + " |"
    lines.append(hdr)
    lines.append(sep)
    for row in str_rows:
        lines.append("| " + " | ".join(cell.ljust(widths[j])
                                         for j, cell in enumerate(row)) + " |")
    return "\n".join(lines)



def build_latex_table(headers: list, rows: list, col_aligns: list = None,
                      bold_map: dict = None, caption: str = "",
                      label: str = "") -> str:
    if bold_map is None:
        bold_map = {}
    n = len(headers)
    if col_aligns is None:
        col_aligns = ["l"] * n
    align_str = "".join(col_aligns)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{align_str}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]

    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            s = _fmt(cell) if not isinstance(cell, str) else cell
            s = s.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")
            if bold_map.get(j) == i:
                s = _bold_tex(s)
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)



MAIN_METHODS = [
    ("vpg",               "VPG"),
    ("vpg_avg",           "VPG + Avg Baseline"),
    ("vpg_value",         "VPG + Value Baseline"),
    ("rloo_K8",           "RLOO K=8"),
    ("vpg_entropy",       "VPG + Entropy"),
    ("vpg_value_entropy", "VPG + Value + Entropy"),
]


def table1_main(grouped: dict) -> tuple:
    headers = ["Method", "Mean Reward (\u2191)", "Std (\u2193)",
               "Solve Episode (\u2193)", "Wall Time (s)", "Grad Var (\u2193)"]
    headers_tex = ["Method", "Mean Reward ($\\uparrow$)", "Std ($\\downarrow$)",
                   "Solve Episode ($\\downarrow$)", "Wall Time (s)",
                   "Grad Var ($\\downarrow$)"]
    aligns = ["l", "r", "r", "r", "r", "r"]
    raw_rows = []

    for key, label in MAIN_METHODS:
        histories = get_histories(grouped, key)
        if not histories:
            raw_rows.append([label, None, None, None, None, None])
            continue

        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        solve_eps = [solve_episode(h) for h in histories]
        solved_vals = [s for s in solve_eps if s is not None]
        times = [h.get("wall_time_total", 0.0) for h in histories]
        gvars = [np.mean(h["grad_variances"][-20:]) for h in histories
                 if h.get("grad_variances")]

        mean_r = np.mean(rewards) if rewards else None
        std_r = np.std(rewards) if len(rewards) > 1 else (0.0 if rewards else None)
        mean_solve = np.mean(solved_vals) if solved_vals else None
        mean_time = np.mean(times) if times else None
        mean_gvar = np.mean(gvars) if gvars else None

        raw_rows.append([label, mean_r, std_r, mean_solve, mean_time, mean_gvar])

    bold_map = {}
    best = _find_best(raw_rows, 1, "max")
    if best is not None:
        bold_map[1] = best
    best = _find_best(raw_rows, 2, "min")
    if best is not None:
        bold_map[2] = best
    best = _find_best(raw_rows, 3, "min")
    if best is not None:
        bold_map[3] = best
    best = _find_best(raw_rows, 5, "min")
    if best is not None:
        bold_map[5] = best

    fmt_rows = []
    for row in raw_rows:
        fmt_rows.append([
            row[0],
            _fmt(row[1], 1),
            _fmt(row[2], 1),
            _fmt(row[3], 0) if row[3] is not None else "\u2014",
            _fmt(row[4], 1),
            f"{row[5]:.2e}" if row[5] is not None else "\u2014",
        ])

    return headers, headers_tex, fmt_rows, aligns, bold_map, "Main Method Comparison"



def table2_rloo(grouped: dict) -> tuple:
    headers = ["K", "Mean Reward", "Std", "Solve Episode"]
    aligns = ["r", "r", "r", "r"]
    fmt_rows = []
    raw_for_bold = []

    for K in [4, 8, 16]:
        key = f"rloo_K{K}"
        histories = get_histories(grouped, key)
        if not histories:
            fmt_rows.append([str(K), "\u2014", "\u2014", "\u2014"])
            raw_for_bold.append([K, None, None, None])
            continue

        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        solve_eps = [solve_episode(h) for h in histories]
        solved_vals = [s for s in solve_eps if s is not None]

        mean_r = np.mean(rewards) if rewards else None
        std_r = np.std(rewards) if len(rewards) > 1 else (0.0 if rewards else None)
        mean_solve = np.mean(solved_vals) if solved_vals else None

        raw_for_bold.append([K, mean_r, std_r, mean_solve])
        fmt_rows.append([
            str(K),
            _fmt(mean_r, 1),
            _fmt(std_r, 1),
            _fmt(mean_solve, 0) if mean_solve is not None else "\u2014",
        ])

    bold_map = {}
    best = _find_best(raw_for_bold, 1, "max")
    if best is not None:
        bold_map[1] = best
    best = _find_best(raw_for_bold, 2, "min")
    if best is not None:
        bold_map[2] = best
    best = _find_best(raw_for_bold, 3, "min")
    if best is not None:
        bold_map[3] = best

    return headers, headers, fmt_rows, aligns, bold_map, "RLOO K Ablation"



def table3_entropy(grouped: dict) -> tuple:
    headers = ["\u03b2", "Schedule", "Final Reward", "Final Entropy", "Solved?"]
    headers_tex = ["$\\beta$", "Schedule", "Final Reward", "Final Entropy", "Solved?"]
    aligns = ["r", "l", "r", "r", "c"]
    fmt_rows = []
    raw_for_bold = []

    for beta_str in ["0.001", "0.01", "0.05", "0.1"]:
        key = f"vpg_entropy_beta{beta_str}"
        histories = get_histories(grouped, key)
        configs = get_configs(grouped, key)
        if not histories:
            continue

        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        entropies = [h["entropies"][-1] for h in histories
                     if h.get("entropies") and len(h["entropies"]) > 0]
        solved_list = [h.get("solved", False) for h in histories]
        schedule = configs[0].get("entropy_schedule", "linear") if configs else "linear"

        mean_r = np.mean(rewards) if rewards else None
        mean_ent = np.mean(entropies) if entropies else None
        n_solved = sum(solved_list)
        total = len(solved_list)

        raw_for_bold.append([float(beta_str), schedule, mean_r, mean_ent, n_solved])
        fmt_rows.append([
            beta_str,
            schedule,
            _fmt(mean_r, 1),
            _fmt(mean_ent, 4),
            f"{n_solved}/{total}",
        ])

    for sched in ["constant", "linear", "cosine"]:
        key = f"vpg_entropy_sched_{sched}"
        histories = get_histories(grouped, key)
        configs = get_configs(grouped, key)
        if not histories:
            continue

        beta = configs[0].get("entropy_beta", 0.01) if configs else 0.01
        rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories
                   if h.get("episode_rewards")]
        entropies = [h["entropies"][-1] for h in histories
                     if h.get("entropies") and len(h["entropies"]) > 0]
        solved_list = [h.get("solved", False) for h in histories]

        mean_r = np.mean(rewards) if rewards else None
        mean_ent = np.mean(entropies) if entropies else None
        n_solved = sum(solved_list)
        total = len(solved_list)

        raw_for_bold.append([beta, sched, mean_r, mean_ent, n_solved])
        fmt_rows.append([
            _fmt(beta, 3),
            sched,
            _fmt(mean_r, 1),
            _fmt(mean_ent, 4),
            f"{n_solved}/{total}",
        ])

    bold_map = {}
    best = _find_best(raw_for_bold, 2, "max")
    if best is not None:
        bold_map[2] = best

    return headers, headers_tex, fmt_rows, aligns, bold_map, "Entropy Ablation"



def table4_bc_size(bc: dict) -> tuple:
    headers = ["Transitions", "Mean Reward", "Std", "Expert Gap"]
    aligns = ["r", "r", "r", "r"]
    fmt_rows = []
    raw_for_bold = []

    size_entries = {
        int(k.replace("size_", "")): v
        for k, v in bc.items()
        if k.startswith("size_")
    }
    if not size_entries:
        return headers, headers, [], aligns, {}, "BC Dataset Size Ablation"

    expert_reward = 500.0
    for n in sorted(size_entries):
        entry = size_entries[n]
        mean_r = entry["mean"]
        std_r = entry["std"]
        gap = expert_reward - mean_r

        raw_for_bold.append([n, mean_r, std_r, gap])
        fmt_rows.append([
            f"{n:,}",
            _fmt(mean_r, 1),
            _fmt(std_r, 1),
            _fmt(gap, 1),
        ])

    bold_map = {}
    best = _find_best(raw_for_bold, 1, "max")
    if best is not None:
        bold_map[1] = best
    best = _find_best(raw_for_bold, 2, "min")
    if best is not None:
        bold_map[2] = best

    return headers, headers, fmt_rows, aligns, bold_map, "BC Dataset Size Ablation"



def table5_bc_noise(bc: dict) -> tuple:
    headers = ["Noise %", "Mean Reward", "Std"]
    aligns = ["r", "r", "r"]
    fmt_rows = []
    raw_for_bold = []

    noise_entries = {
        float(k.replace("noise_", "")): v
        for k, v in bc.items()
        if k.startswith("noise_")
    }
    if not noise_entries:
        return headers, headers, [], aligns, {}, "BC Noise Ablation"

    for ratio in sorted(noise_entries):
        entry = noise_entries[ratio]
        mean_r = entry["mean"]
        std_r = entry["std"]

        raw_for_bold.append([ratio, mean_r, std_r])
        fmt_rows.append([
            f"{ratio * 100:.0f}%",
            _fmt(mean_r, 1),
            _fmt(std_r, 1),
        ])

    bold_map = {}
    best = _find_best(raw_for_bold, 1, "max")
    if best is not None:
        bold_map[1] = best

    return headers, headers, fmt_rows, aligns, bold_map, "BC Noise Ablation"



ALL_TABLES = {
    1: ("Table 1: Main Comparison",     table1_main),
    2: ("Table 2: RLOO K Ablation",     table2_rloo),
    3: ("Table 3: Entropy Ablation",    table3_entropy),
    4: ("Table 4: BC Dataset Size",     table4_bc_size),
    5: ("Table 5: BC Noise Ablation",   table5_bc_noise),
}

_USES_GROUPED = {1, 2, 3}
_USES_BC = {4, 5}


def generate_all(grouped: dict, bc: dict, fmt: str) -> str:
    sections = []

    for num, (title, build_fn) in ALL_TABLES.items():
        if num in _USES_GROUPED:
            result = build_fn(grouped)
        else:
            result = build_fn(bc)

        headers_md, headers_tex, fmt_rows, aligns, bold_map, subtitle = result

        if not fmt_rows:
            sections.append(f"### {title}\n\n*(no data)*\n")
            continue

        if fmt in ("markdown", "both"):
            md = build_markdown_table(headers_md, fmt_rows, aligns, bold_map)
            sections.append(f"### {title}\n\n{md}\n")

        if fmt in ("latex", "both"):
            label = f"tab{num}"
            tex = build_latex_table(headers_tex, fmt_rows, aligns, bold_map,
                                    caption=subtitle, label=label)
            if fmt == "both":
                sections.append(f"```latex\n{tex}\n```\n")
            else:
                sections.append(f"% {title}\n{tex}\n")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="markdown")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "raw"))
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...", file=sys.stderr)
    grouped = load_grouped(args.results_dir)
    bc = load_bc(args.results_dir)
    print(f"  Methods: {sorted(grouped.keys())}", file=sys.stderr)
    print(f"  BC data: {'yes' if bc else 'no'}", file=sys.stderr)

    output = generate_all(grouped, bc, args.format)
    print(output)

    ext = "md" if args.format == "markdown" else ("tex" if args.format == "latex" else "md")
    out_path = os.path.join(os.path.dirname(args.results_dir), f"tables_{args.format}.{ext}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
