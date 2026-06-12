import os
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from contextlib import contextmanager

# Okabe-Ito palette
PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#CC79A7',  # pink
    '#F0E442',  # yellow
    '#000000',  # black
]

C = {
    'blue': PALETTE[0], 'red': PALETTE[1], 'green': PALETTE[2],
    'orange': PALETTE[3], 'cyan': PALETTE[4], 'purple': PALETTE[5],
    'yellow': PALETTE[6], 'black': PALETTE[7],
}


@contextmanager
def _clean_style():
    with plt.rc_context({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    }):
        yield


def _save(fig, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_accuracy_bars(results_dict, difficulties, save_path):
    """Grouped bar chart of iso/non-iso accuracy by difficulty for N models.

    results_dict: {'Model': {diff: {'iso': float, 'non_iso': float, 'floor': float}}}
    """
    models = list(results_dict.keys())
    n_models = len(models)
    n_diffs = len(difficulties)

    n_bars = 2 * n_models
    width = 0.8 / n_bars
    x = np.arange(n_diffs)

    with _clean_style():
        fig, ax = plt.subplots(figsize=(max(8, n_diffs * 1.2), 6))

        bar_colors = [
            (C['blue'], C['cyan']),
            (C['red'], C['orange']),
            (C['green'], C['yellow']),
            (C['purple'], C['black']),
        ]

        for m_idx, model in enumerate(models):
            iso_vals = [results_dict[model][d]['iso'] for d in difficulties]
            non_vals = [results_dict[model][d]['non_iso'] for d in difficulties]
            c_iso, c_non = bar_colors[m_idx % len(bar_colors)]

            offset_iso = (2 * m_idx - n_bars / 2 + 0.5) * width
            offset_non = (2 * m_idx + 1 - n_bars / 2 + 0.5) * width

            ax.bar(x + offset_iso, iso_vals, width, label=f'{model} Iso',
                   color=c_iso, edgecolor='white', linewidth=0.5)
            ax.bar(x + offset_non, non_vals, width, label=f'{model} Non-Iso',
                   color=c_non, edgecolor='white', linewidth=0.5)

        # Floor line (take from first model — floor is dataset-level, not model-level)
        first = models[0]
        floors = [results_dict[first][d].get('floor', 0) for d in difficulties]
        ax.plot(x, floors, color=C['black'], ls='--', marker='s', ms=5,
                label='Degenerate Floor', zorder=5)

        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Difficulty')
        ax.set_title('Accuracy by Difficulty')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.0))

        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_training_curves(metrics_csv_paths, metric_name, save_path,
                         smooth_window=5):
    """Step vs metric for multiple runs. smooth_window=0 disables EMA."""
    with _clean_style():
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, (run_name, csv_path) in enumerate(metrics_csv_paths.items()):
            if not os.path.exists(csv_path):
                print(f"[plots] Skipping {run_name}: {csv_path} not found")
                continue

            steps, vals = [], []
            with open(csv_path, 'r') as f:
                for row in csv.DictReader(f):
                    if metric_name in row and row.get(metric_name, '') != '':
                        try:
                            s = float(row.get('step', row.get('global_step', 0)))
                            v = float(row[metric_name])
                            steps.append(s)
                            vals.append(v)
                        except (ValueError, TypeError):
                            pass

            if not steps:
                continue

            color = PALETTE[idx % len(PALETTE)]
            steps, vals = np.array(steps), np.array(vals)

            ax.plot(steps, vals, color=color, alpha=0.25, lw=1)

            # EMA smoothing
            if smooth_window > 0 and len(vals) > smooth_window:
                alpha_ema = 2.0 / (smooth_window + 1)
                smoothed = np.zeros_like(vals)
                smoothed[0] = vals[0]
                for i in range(1, len(vals)):
                    smoothed[i] = alpha_ema * vals[i] + (1 - alpha_ema) * smoothed[i - 1]
                ax.plot(steps, smoothed, color=color, lw=2, label=run_name)
            else:
                ax.plot(steps, vals, color=color, lw=2, label=run_name)

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(metric_name.replace("_", " ").title())
        ax.legend()
        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_collapse_monitor(collapse_csv_path, save_path):
    """Dual y-axis: NOT-ISO ratio (left) and frac_zero_std (right) over steps."""
    if not os.path.exists(collapse_csv_path):
        raise FileNotFoundError(f"{collapse_csv_path} not found")

    steps, ratios, fracs = [], [], []
    with open(collapse_csv_path, 'r') as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row['step']))
                ratios.append(float(row['class_prediction_ratio']))
                fracs.append(float(row['frac_zero_std']))
            except (ValueError, KeyError):
                pass

    with _clean_style():
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(steps, ratios, color=C['red'], lw=2, label='NOT-ISO ratio')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('NOT-ISO ratio', color=C['red'])
        ax1.tick_params(axis='y', labelcolor=C['red'])
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        ax1.axhline(0.80, color='gray', ls=':', lw=1, alpha=0.7)
        ax1.axhline(0.95, color=C['black'], ls='--', lw=1.5, alpha=0.8)
        ax1.annotate('Warning (80%)', xy=(steps[-1] if steps else 0, 0.80),
                     fontsize=8, color='gray', va='bottom', ha='right')
        ax1.annotate('Collapse (95%)', xy=(steps[-1] if steps else 0, 0.95),
                     fontsize=8, color=C['black'], va='bottom', ha='right')

        ax2 = ax1.twinx()
        ax2.plot(steps, fracs, color=C['blue'], lw=2, ls='-.', label='frac_zero_std')
        ax2.set_ylabel('frac_zero_std', color=C['blue'])
        ax2.tick_params(axis='y', labelcolor=C['blue'])
        ax2.set_ylim(0, 1.05)
        ax2.spines['right'].set_visible(True)  # Need right spine for twin axis

        ax1.set_title('collapse monitor')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')

        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_error_taxonomy(error_dicts, model_names, save_path):
    """Stacked bar chart of 4 error categories per model."""
    categories = ['err_format_fail', 'err_wrong_mapping',
                  'err_false_not_iso', 'err_false_mapping_claim']
    labels = ['Format Fail', 'Wrong Mapping',
              'False "NOT ISO"', 'False Mapping Claim']
    colors = [C['purple'], C['orange'], C['red'], C['blue']]

    with _clean_style():
        fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 6))
        x = np.arange(len(model_names))
        width = 0.5
        bottom = np.zeros(len(model_names))

        for cat, label, color in zip(categories, labels, colors):
            vals = np.array([ed.get(cat, 0) for ed in error_dicts], dtype=float)
            bars = ax.bar(x, vals, width, bottom=bottom, label=label,
                          color=color, edgecolor='white', linewidth=0.5)
            for bar, v, b in zip(bars, vals, bottom):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                            f'{int(v)}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
            bottom += vals

        ax.set_ylabel('Count')
        ax.set_title('Error Taxonomy')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left')

        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_cot_length_histogram(lengths_before, lengths_after, save_path,
                              label_before='Base Model',
                              label_after='Trained Model'):
    """Overlaid histograms of <think> section lengths with median markers."""
    lengths_before = np.asarray(lengths_before, dtype=float)
    lengths_after = np.asarray(lengths_after, dtype=float)

    with _clean_style():
        fig, ax = plt.subplots(figsize=(10, 5))

        all_vals = np.concatenate([lengths_before, lengths_after])
        bins = np.linspace(all_vals.min(), all_vals.max(), 35)

        ax.hist(lengths_before, bins=bins, color=C['blue'], alpha=0.45,
                label=label_before, edgecolor=C['blue'], linewidth=0.8)
        ax.hist(lengths_after, bins=bins, color=C['red'], alpha=0.45,
                label=label_after, edgecolor=C['red'], linewidth=0.8)

        med_b = np.median(lengths_before)
        med_a = np.median(lengths_after)
        ax.axvline(med_b, color=C['blue'], ls='--', lw=1.5,
                   label=f'{label_before} median ({med_b:.0f})')
        ax.axvline(med_a, color=C['red'], ls='--', lw=1.5,
                   label=f'{label_after} median ({med_a:.0f})')

        ax.set_xlabel('think length (chars)')
        ax.set_ylabel('count')
        ax.set_title('CoT length')
        ax.legend()

        fig.tight_layout()
        _save(fig, save_path)
    return fig
