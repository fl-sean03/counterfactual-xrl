"""Generate explanations for every DecisionRecord in a directory.

Usage:
    python scripts/explain.py --records-dir results/decision_records \
        --out-dir results/explanations
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from xrl.analysis.records import load_record
from xrl.explainer.client import make_client
from xrl.explainer.pipeline import explain
from xrl.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out-dir", default="results/explanations")
    ap.add_argument("--config", default="configs/explainer.yaml")
    ap.add_argument("--force-mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap number of records (debug)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg["generator"]
    log_dir = Path(cfg.get("log_dir", "results/explanations/logs"))
    cache_dir = Path(cfg.get("cache_dir", "results/explanations/cache"))

    client = make_client(
        mock=True if args.force_mock else None,
        model=gen_cfg["model"],
        cost_cap_usd=gen_cfg["cost_cap_usd"],
        log_dir=log_dir / "generator",
        provider=cfg.get("provider"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_paths = sorted(Path(args.records_dir).rglob("*.json"))
    if args.limit:
        rec_paths = rec_paths[: args.limit]

    written = 0
    for p in rec_paths:
        record = load_record(p)
        exp = explain(record, client, cache_dir=cache_dir)
        rel = p.relative_to(args.records_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(dst, "w") as f:
            json.dump(asdict(exp), f, indent=2)
        written += 1

    total_cost = getattr(client, "total_cost", 0.0)
    print(f"Wrote {written} explanations ({client.calls} API calls, ${total_cost:.4f} total)")


if __name__ == "__main__":
    main()
