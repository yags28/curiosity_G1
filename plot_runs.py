"""
Compare curiosity method training runs from local CSV logs.

Usage:
  python3 plot_runs.py                          # auto-detect all runs in logs/
  python3 plot_runs.py --runs logs/rnd logs/drnd logs/rdd
  python3 plot_runs.py --task distant_target    # filter by task prefix
  python3 plot_runs.py --smooth 20              # rolling-window smoothing (default 10)
  python3 plot_runs.py --out compare.png        # save instead of show
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# ── minimal deps check ────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("Install matplotlib: pip install matplotlib numpy")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> dict[str, list]:
    data: dict[str, list] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid").tolist()


def find_runs(logs_root: Path, task_filter: str | None) -> list[Path]:
    """Return sorted list of dirs under logs/ that contain metrics.csv."""
    runs = []
    for d in sorted(logs_root.iterdir()):
        csv = d / "metrics.csv"
        if d.is_dir() and csv.exists():
            if task_filter and task_filter not in d.name:
                continue
            runs.append(d)
    return runs


# ── palette (colourblind-friendly) ───────────────────────────────────────────

_COLORS = {
    "rnd":  "#0077BB",
    "drnd": "#EE7733",
    "rdd":  "#009988",
}

def _color(run_name: str) -> str:
    for key, col in _COLORS.items():
        if key in run_name.lower():
            return col
    return None  # matplotlib picks automatically


# ── main ─────────────────────────────────────────────────────────────────────

METRICS = [
    ("episode_reward",  "Episode Reward"),
    ("success_rate",    "Success Rate"),
    ("intrinsic_reward","Intrinsic Reward"),
    ("curiosity_loss",  "Curiosity Loss"),
    ("policy_loss",     "Policy Loss"),
    ("entropy",         "Entropy"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="*", default=None,
                   help="Explicit run dirs. Default: auto-detect from logs/")
    p.add_argument("--task",   default=None, help="Filter runs by task name substring")
    p.add_argument("--smooth", type=int, default=10, help="Rolling-average window")
    p.add_argument("--out",    default=None, help="Save path (e.g. compare.png)")
    args = p.parse_args()

    logs_root = Path("logs")
    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        if not logs_root.exists():
            sys.exit("No logs/ directory found. Run at least one training first.")
        run_dirs = find_runs(logs_root, args.task)

    if not run_dirs:
        sys.exit("No runs found. Check your --runs paths or --task filter.")

    # Load data
    runs: dict[str, dict] = {}
    for d in run_dirs:
        csv_path = d / "metrics.csv"
        if not csv_path.exists():
            print(f"[warn] no metrics.csv in {d}, skipping")
            continue
        runs[d.name] = load_csv(csv_path)
        print(f"  loaded {d.name}  ({len(runs[d.name].get('global_step', []))} rows)")

    if not runs:
        sys.exit("No valid runs loaded.")

    # Plot
    nrows = (len(METRICS) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, METRICS):
        for name, data in runs.items():
            if col not in data:
                continue
            xs = data["global_step"]
            ys = smooth(data[col], args.smooth)
            # align xs after smoothing (valid convolution shortens the array)
            xs_plot = xs[len(xs) - len(ys):]
            ax.plot(xs_plot, ys, label=name, color=_color(name), linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide any unused subplot slots
    for ax in axes[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle(f"Curiosity Method Comparison — smooth={args.smooth}", fontsize=13)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
