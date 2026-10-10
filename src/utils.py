"""Shared utilities: seeding, time parsing/formatting, JSON I/O."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the pipeline for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_time_to_seconds(value, event: str) -> float | None:
    """Parse a performance into seconds.

    Accepts floats, "10.85" (sprint) and "1:52.30" (800m-style) strings.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        if ":" in s:
            minutes, rest = s.split(":", 1)
            return float(minutes) * 60.0 + float(rest)
        return float(s)
    except ValueError:
        return None


def format_time(seconds, event: str) -> str:
    """Format seconds for display: "10.85" or "1:52.30"."""
    if seconds is None:
        return "—"
    try:
        if np.isnan(float(seconds)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    seconds = float(seconds)
    if event in ("800m", "1500m"):
        minutes = int(seconds // 60.0)
        rest = seconds - 60.0 * minutes
        return f"{minutes}:{rest:05.2f}"
    return f"{seconds:.2f}"


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)
