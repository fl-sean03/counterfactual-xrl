"""Regenerate every figure and table referenced in the final report.

All figures are saved to ``report/figures/``. Data sources are read from
``results/``, if a source is missing the corresponding figure is
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
    """Bar chart: success rate with 95% bootstrap CI for each agent.

    DQN is intentionally excluded from the headline figure: the three DQN
    variants collapse to a stall policy on this environment and are
    reported separately in the "Why DQN was cut" ablation.
    """
    rows = []
    runs = [
        ("Random", ROOT / "results/random/eval_summary.json"),
        ("PPO (tuned)", ROOT / "results/ppo/tuned/seed0/eval_summary.json"),
        ("MCTS", ROOT / "results/mcts/baseline/seed0/eval_summary.json"),
    ]
    if all(p.exists() for _, p in runs):
        for label, p in runs:
            with open(p) as f:
                d = json.load(f)
            lo, hi = d["success"]["ci"]
            rows.append({"agent": label, "success": d["success"]["mean"], "lo": lo, "hi": hi})
    else:
        csv_path = FIG_DIR / "eval_summary.csv"
        if not csv_path.exists():
            print("fig_task_performance: no eval data, skipping")
            return
        df_csv = pd.read_csv(csv_path)
        for _, r in df_csv.iterrows():
            ci = r["Success CI"].strip("[]").split(",")
            rows.append(
                {
                    "agent": r["Agent"],
                    "success": float(r["Success"]),
                    "lo": float(ci[0]),
                    "hi": float(ci[1]),
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["success"], color="#4C72B0", width=0.55)
    err_lo = df["success"] - df["lo"]
    err_hi = df["hi"] - df["success"]
    ax.errorbar(
        xs, df["success"], yerr=[err_lo, err_hi], fmt="none", ecolor="k", capsize=4, lw=1.0
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(df["agent"], rotation=15, ha="right")
    ax.set_ylabel("Success rate over evaluation episodes")
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Task performance (95% bootstrap CI)", pad=14)
    for bar, v in zip(bars, df["success"], strict=False):
        text = f"{v:.3f}" if v < 0.05 else f"{v:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.05,
            text,
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, "task_performance")
    plt.close(fig)


def fig_metric_comparison() -> None:
    """Two-panel: mismatched-performance run vs matched-performance run.

    Shows the collapse of the MCTS-tree-vs-PPO-rollout explanation gap
    once both agents are competent. Mismatched-run numbers are the
    final values logged from the 0.59-task-success PPO baseline; matched
    numbers are the values from the tuned-PPO performance-match.
    """
    panels = [
        {
            "title": "Mismatched (PPO 0.590 success)",
            "ppo": {
                "fidelity":    (0.899, 0.880, 0.920),
                "soundness":   (0.847, 0.820, 0.870),
                "inferability":(0.959, 0.920, 0.990),
            },
            "mcts": {
                "fidelity":    (0.928, 0.910, 0.950),
                "soundness":   (0.879, 0.860, 0.900),
                "inferability":(0.973, 0.950, 0.990),
            },
        },
        {
            "title": "Matched (PPO 1.000 success)",
            "ppo": {
                "fidelity":    (0.924, 0.906, 0.939),
                "soundness":   (0.884, 0.862, 0.906),
                "inferability":(0.979, 0.950, 1.000),
            },
            "mcts": {
                "fidelity":    (0.924, 0.907, 0.940),
                "soundness":   (0.892, 0.872, 0.913),
                "inferability":(0.986, 0.966, 1.000),
            },
        },
    ]
    metrics = ["fidelity", "soundness", "inferability"]
    metric_labels = ["Fidelity", "Soundness", "Inferability"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    width = 0.36
    color_mcts = "#4C72B0"
    color_ppo = "#DD8452"
    x = np.arange(len(metrics))
    for ax, panel in zip(axes, panels, strict=False):
        for i, (src_key, src_label, color, offset) in enumerate(
            [
                ("mcts", "MCTS tree",   color_mcts, -width / 2),
                ("ppo",  "PPO rollout", color_ppo,  +width / 2),
            ]
        ):
            vals = [panel[src_key][m][0] for m in metrics]
            lo = [panel[src_key][m][0] - panel[src_key][m][1] for m in metrics]
            hi = [panel[src_key][m][2] - panel[src_key][m][0] for m in metrics]
            ax.bar(x + offset, vals, width, color=color, label=src_label)
            ax.errorbar(
                x + offset, vals, yerr=[lo, hi], fmt="none",
                ecolor="k", capsize=3, lw=1.0,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0.78, 1.03)
        ax.set_yticks(np.linspace(0.80, 1.00, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(panel["title"], pad=8)
    axes[0].set_ylabel("Score (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Explanation quality: MCTS tree vs. PPO rollout (95% bootstrap CI)",
        y=1.02,
    )
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "metric_comparison")
    plt.close(fig)


def fig_learning_curves() -> None:
    """PPO learning curve from monitor CSVs (one per parallel env)."""
    ppo_dir = ROOT / "results/ppo/tuned/seed0"
    monitor_csvs = sorted(ppo_dir.glob("monitor_*.csv"))
    if not monitor_csvs:
        print("fig_learning_curves: no PPO monitor*.csv found, skipping")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    all_returns: list[pd.Series] = []
    for p in monitor_csvs:
        try:
            df = pd.read_csv(p, skiprows=1)
        except Exception:
            continue
        if "r" not in df.columns:
            continue
        all_returns.append(df["r"].reset_index(drop=True))
    if not all_returns:
        print("fig_learning_curves: PPO monitor*.csv had no returns column")
        plt.close(fig)
        return
    combined = pd.concat(all_returns, ignore_index=True)
    window = max(1, len(combined) // 60)
    smoothed = combined.rolling(window, min_periods=1).mean()
    ax.plot(smoothed.to_numpy(), color="#4C72B0", alpha=0.9, label="PPO (tuned)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Training episode (across all parallel envs)")
    ax.set_ylabel("Smoothed episode return")
    ax.set_title("PPO learning curve (rolling mean)")
    ax.legend(frameon=False)
    _save(fig, "learning_curves")
    plt.close(fig)


def _collect_eval_rows(run_pairs: list[tuple[str, Path]]) -> list[dict]:
    rows = []
    for label, p in run_pairs:
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
    return rows


def table_eval_summary() -> None:
    """Headline table: Random / PPO / MCTS only. DQN is in the ablation
    table (see ``ablation_summary.csv``).
    """
    rows = _collect_eval_rows(
        [
            ("Random", ROOT / "results/random/eval_summary.json"),
            ("PPO (tuned)", ROOT / "results/ppo/tuned/seed0/eval_summary.json"),
            ("MCTS", ROOT / "results/mcts/baseline/seed0/eval_summary.json"),
        ]
    )
    if not rows:
        return
    out = FIG_DIR / "eval_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out}")


def table_ablation_summary() -> None:
    """DQN ablation table referenced from the report's "Why DQN was cut"
    paragraph. Skipped silently if the DQN runs are not present.
    """
    rows = _collect_eval_rows(
        [
            ("DQN (image)", ROOT / "results/dqn/baseline/seed0/eval_summary.json"),
            ("DQN (symbolic)", ROOT / "results/dqn/symbolic/seed0/eval_summary.json"),
            ("DQN (shaped)", ROOT / "results/dqn/shaped/seed0/eval_summary.json"),
        ]
    )
    if not rows:
        return
    out = FIG_DIR / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out}")


def main() -> None:
    print("Regenerating report artifacts in", FIG_DIR)
    fig_task_performance()
    fig_learning_curves()
    fig_metric_comparison()
    table_eval_summary()
    table_ablation_summary()


if __name__ == "__main__":
    main()
