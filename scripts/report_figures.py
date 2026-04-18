"""Regenerate every figure and table referenced in the final report.

All figures are saved to ``report/figures/``. Data sources are read from
``results/`` — if a source is missing the corresponding figure is
skipped with a warning (so this script is safe to run at any stage).

Usage:
    python scripts/report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    print(f"  wrote {path}")


def fig_task_performance() -> None:
    """Bar chart: success rate with 95% bootstrap CI for each agent."""
    rows = []
    runs = [
        ("Random", ROOT / "results/random/eval_summary.json"),
        ("DQN (image)", ROOT / "results/dqn/baseline/seed0/eval_summary.json"),
        ("DQN (symbolic)", ROOT / "results/dqn/symbolic/seed0/eval_summary.json"),
        ("DQN (shaped)", ROOT / "results/dqn/shaped/seed0/eval_summary.json"),
        ("MCTS", ROOT / "results/mcts/baseline/seed0/eval_summary.json"),
    ]
    for label, p in runs:
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        lo, hi = d["success"]["ci"]
        rows.append({"agent": label, "success": d["success"]["mean"], "lo": lo, "hi": hi})
    if not rows:
        print("fig_task_performance: no eval summaries yet, skipping")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["success"], color="#4C72B0")
    err_lo = df["success"] - df["lo"]
    err_hi = df["hi"] - df["success"]
    ax.errorbar(xs, df["success"], yerr=[err_lo, err_hi], fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["agent"], rotation=20, ha="right")
    ax.set_ylabel("Success rate over 300 episodes")
    ax.set_ylim(0, 1)
    ax.set_title("Task performance (95% bootstrap CI)")
    for bar, v in zip(bars, df["success"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    _save(fig, "task_performance")
    plt.close(fig)


def fig_metric_comparison() -> None:
    """Grouped bar: fidelity/soundness/inferability for each source."""
    p = ROOT / "results/metrics/summary.json"
    if not p.exists():
        print("fig_metric_comparison: no metrics summary yet, skipping")
        return
    with open(p) as f:
        summary = json.load(f)
    sources = list(summary.keys())
    if not sources:
        return
    metrics = ["fidelity", "soundness", "inferability"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(sources))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, src in enumerate(sources):
        vals = [summary[src][m]["mean"] for m in metrics]
        cis = [summary[src][m]["ci"] for m in metrics]
        lo = [v - c[0] for v, c in zip(vals, cis, strict=False)]
        hi = [c[1] - v for v, c in zip(vals, cis, strict=False)]
        offset = (i - (len(sources) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=src, color=colors[i % len(colors)])
        ax.errorbar(x + offset, vals, yerr=[lo, hi], fmt="none", ecolor="k", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Explanation quality by evidence source")
    _save(fig, "metric_comparison")
    plt.close(fig)


def fig_learning_curves() -> None:
    """Monitor CSV learning curves, smoothed."""
    paths = [
        ("DQN image", ROOT / "results/dqn/baseline/seed0/monitor.csv"),
        ("DQN symbolic", ROOT / "results/dqn/symbolic/seed0/monitor.csv"),
        ("DQN shaped", ROOT / "results/dqn/shaped/seed0/monitor.csv"),
    ]
    any_plotted = False
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for label, p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, skiprows=1)
        except Exception:
            continue
        if "r" not in df.columns:
            continue
        r = df["r"].to_numpy()
        window = max(1, len(r) // 50)
        smoothed = pd.Series(r).rolling(window, min_periods=1).mean()
        ax.plot(smoothed, label=label, alpha=0.8)
        any_plotted = True
    if not any_plotted:
        print("fig_learning_curves: no monitor.csv found, skipping")
        plt.close(fig)
        return
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Smoothed episode return")
    ax.set_title("DQN learning curves (rolling mean)")
    ax.legend(frameon=False)
    _save(fig, "learning_curves")
    plt.close(fig)


def table_eval_summary() -> None:
    """Markdown table the report can include via LaTeX import."""
    rows = []
    runs = [
        ("Random", ROOT / "results/random/eval_summary.json"),
        ("DQN (image)", ROOT / "results/dqn/baseline/seed0/eval_summary.json"),
        ("DQN (symbolic)", ROOT / "results/dqn/symbolic/seed0/eval_summary.json"),
        ("DQN (shaped)", ROOT / "results/dqn/shaped/seed0/eval_summary.json"),
        ("MCTS", ROOT / "results/mcts/baseline/seed0/eval_summary.json"),
    ]
    for label, p in runs:
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append(
            {
                "Agent": label,
                "Success": f"{d['success']['mean']:.3f}",
                "Success CI": f"[{d['success']['ci'][0]:.3f}, {d['success']['ci'][1]:.3f}]",
                "Collision": f"{d['collision']['mean']:.3f}",
                "Return": f"{d['return_']['mean']:.3f}",
                "Steps": f"{d['steps']['mean']:.1f}",
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = FIG_DIR / "eval_summary.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}")


def main() -> None:
    print("Regenerating report artifacts in", FIG_DIR)
    fig_task_performance()
    fig_learning_curves()
    fig_metric_comparison()
    table_eval_summary()


if __name__ == "__main__":
    main()
