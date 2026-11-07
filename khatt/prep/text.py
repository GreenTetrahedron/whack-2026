"""Name normalisation for shop de-duplication (Arabic + Latin), CLAUDE.md §6 (04)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

TASHKEEL = re.compile("[ً-ْ]")
TATWEEL = "ـ"
ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا",
                               "ة": "ه", "ى": "ي"})
SPACES = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    """Strip tashkeel/tatweel; أ إ آ → ا; ة → ه; ى → ي; lowercase Latin; collapse whitespace."""
    if not name:
        return ""
    s = TASHKEEL.sub("", str(name)).replace(TATWEEL, "")
    s = s.translate(ALEF_VARIANTS).lower()
    return SPACES.sub(" ", s).strip()


def similarity(a: str, b: str) -> float:
    """0..1 on already-normalised names; empty names never match."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
