"""Score a set of explanations against their decision records.

Usage:
    python scripts/evaluate_explanations.py \
        --records-dir results/decision_records \
        --explanations-dir results/explanations \
        --out-dir results/metrics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from xrl.analysis.records import load_record
from xrl.eval.metrics import aggregate, score_pair
from xrl.explainer.client import make_client
from xrl.explainer.pipeline import Explanation
from xrl.utils.config import load_config


def load_explanation(path: Path) -> Explanation:
    with open(path) as f:
        d = json.load(f)
    return Explanation(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--explanations-dir", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--config", default="configs/explainer.yaml")
    ap.add_argument("--force-mock", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    judge = make_client(
        mock=True if args.force_mock else None,
        model=jcfg["model"],
        cost_cap_usd=jcfg["cost_cap_usd"],
        log_dir=Path(cfg.get("log_dir", "results/explanations/logs")) / "judge",
        provider=cfg.get("provider"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_dir = Path(args.records_dir)
    expl_dir = Path(args.explanations_dir)

    per_record_rows: list[dict] = []
    by_source: dict[str, list] = {}

    for rec_path in sorted(records_dir.rglob("*.json")):
        rel = rec_path.relative_to(records_dir)
        exp_path = expl_dir / rel
        if not exp_path.exists():
            continue
        record = load_record(rec_path)
        exp = load_explanation(exp_path)
        scores = score_pair(record, exp, judge)
        per_record_rows.append(
            {
                "agent_id": record.agent_id,
                "source": record.source,
                "state_id": record.state_id,
                "step": record.step,
                **asdict(scores),
            }
        )
        by_source.setdefault(record.source, []).append(scores)

    df = pd.DataFrame(per_record_rows)
    df.to_csv(out_dir / "per_record.csv", index=False)

    summary = {src: aggregate(lst) for src, lst in by_source.items()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Scored {len(per_record_rows)} records. Judge calls: {judge.calls}")
    for src, s in summary.items():
        print(f"  {src}: n={s['n']}")
        for k in ("fidelity", "soundness", "inferability"):
            lo, hi = s[k]["ci"]
            print(f"    {k:>14s}: {s[k]['mean']:.3f}   95%CI=({lo:.3f}, {hi:.3f})")
    if hasattr(judge, "total_cost"):
        print(f"  Judge total cost: ${judge.total_cost:.4f}")


if __name__ == "__main__":
    main()
