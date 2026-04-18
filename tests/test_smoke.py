"""Phase 0 baseline tests — package imports and seeding work."""

from __future__ import annotations


def test_package_imports() -> None:
    import xrl

    assert xrl.__version__


def test_seed_everything_runs() -> None:
    from xrl.utils.seeding import seed_everything

    seed_everything(42)
