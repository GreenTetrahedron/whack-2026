"""Shared plumbing for pipeline/ scripts: paths, logging, summaries, DECISIONS.md."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from khatt.config import ROOT, Params, data_dir, processed_dir

RAW = data_dir() / "raw"
INTERIM = data_dir() / "interim"
PROCESSED = processed_dir()
DECISIONS = ROOT / "DECISIONS.md"


def setup(step: str) -> tuple[logging.Logger, Params, float]:
    for d in (RAW, INTERIM, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    log = logging.getLogger(step)
    return log, Params.load(calibrated=False), time.perf_counter()


def summary(step: str, lines: list[str], t0: float) -> None:
    """The 5-line summary every pipeline step prints (CLAUDE.md §19)."""
    print(f"\n=== {step} summary ===")
    for line in lines[:4]:
        print(f"  {line}")
    print(f"  runtime: {time.perf_counter() - t0:.1f} s")


def decision(step: str, text: str) -> None:
    """Append a notable choice to DECISIONS.md (idempotent per identical line)."""
    entry = f"- **{step}**: {text}"
    existing = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else ""
    if entry in existing:
        return
    with open(DECISIONS, "a", encoding="utf-8") as fh:
        if "## Pipeline runs" not in existing:
            fh.write("\n## Pipeline runs\n\n")
        fh.write(f"{entry} _({datetime.now():%Y-%m-%d %H:%M})_\n")


def require(path: Path, how: str) -> Path:
    """Stop with a clear message if an input is missing (never silently continue)."""
    if not Path(path).exists():
        print(f"\nMISSING INPUT: {path}\n  {how}\n", file=sys.stderr)
        sys.exit(2)
    return Path(path)
