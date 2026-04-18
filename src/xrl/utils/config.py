"""Minimal YAML config loader with seed/run-dir conveniences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_run_dir(run_dir: str | Path) -> Path:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_meta(run_dir: str | Path, meta: dict[str, Any]) -> None:
    p = Path(run_dir) / "meta.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, default=str)
