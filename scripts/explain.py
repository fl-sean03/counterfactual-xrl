"""Generate explanations for every DecisionRecord in a directory.

Usage:
    python scripts/explain.py --records-dir results/decision_records \
        --out-dir results/explanations \
        [--workers 8]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from xrl.analysis.records import load_record
from xrl.explainer.client import make_client
from xrl.explainer.pipeline import explain
from xrl.utils.config import load_config


def _process_one(p: Path, records_dir: Path, out_dir: Path, client, cache_dir: Path):
    record = load_record(p)
    exp = explain(record, client, cache_dir=cache_dir)
    rel = p.relative_to(records_dir)
    dst = out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(asdict(exp), f, indent=2)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out-dir", default="results/explanations")
    ap.add_argument("--config", default="configs/explainer.yaml")
    ap.add_argument("--force-mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap number of records (debug)")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel API calls (gpt-4o is thread-safe via the openai SDK)",
    )
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

    records_dir = Path(args.records_dir)
    rec_paths = sorted(records_dir.rglob("*.json"))
    if args.limit:
        rec_paths = rec_paths[: args.limit]

    written = 0
    failures: list[tuple[str, str]] = []

    n_workers = max(1, int(args.workers))
    if args.force_mock or n_workers == 1:
        # Serial path keeps the test/mock flow deterministic.
        for p in rec_paths:
            try:
                _process_one(p, records_dir, out_dir, client, cache_dir)
                written += 1
            except Exception as e:  # noqa: BLE001
                failures.append((str(p), f"{type(e).__name__}: {e}"))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one, p, records_dir, out_dir, client, cache_dir): p
                for p in rec_paths
            }
            for i, fut in enumerate(as_completed(futures), 1):
                p = futures[fut]
                try:
                    fut.result()
                    written += 1
                except Exception as e:  # noqa: BLE001
                    failures.append((str(p), f"{type(e).__name__}: {e}"))
                if i % 20 == 0:
                    print(
                        f"  {i}/{len(rec_paths)} done "
                        f"(written={written}, failed={len(failures)})",
                        flush=True,
                    )

    total_cost = getattr(client, "total_cost", 0.0)
    print(f"Wrote {written} explanations ({client.calls} API calls, ${total_cost:.4f} total)")
    if failures:
        print(f"Skipped {len(failures)} records due to errors:")
        for path, err in failures[:10]:
            print(f"  {path}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
