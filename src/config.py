"""Configuration loading with defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "event": "100m",
    "seed": 42,
    "data": {
        "source": "auto",
        "n_synthetic_athletes": 2400,
        "cache_dir": "data/raw",
        "processed_dir": "data/processed",
        "min_seasons": 4,
    },
    "features": {"history_len": 10, "horizons": 5, "min_history": 3},
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "model": {
        "kind": "lstm",
        "epochs": 40,
        "batch_size": 128,
        "lr": 0.002,
        "hidden": 64,
        "num_layers": 1,
        "nhead": 4,
        "ff_dim": 128,
        "dropout": 0.1,
        "quantiles": [0.1, 0.5, 0.9],
        "early_stop_patience": 8,
    },
    "clustering": {"k": 4},
    "demo": {"host": "127.0.0.1", "port": 8000},
    "artifacts_dir": "artifacts",
    "reports_dir": "reports",
}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load YAML config and deep-merge user overrides on top of defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, file_cfg)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
