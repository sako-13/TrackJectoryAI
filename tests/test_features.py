"""Tests for feature engineering and sequence building."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.features import (
    FEATURE_COLUMNS,
    build_athlete_sequences,
    build_training_samples,
)


def test_build_athlete_sequences(summary):
    seqs = build_athlete_sequences(summary)
    assert len(seqs) > 0
    for seq in seqs.values():
        assert seq.features.shape == (seq.T, len(FEATURE_COLUMNS))
        assert seq.ages.shape == seq.season_best.shape
        # chronological order
        assert (np.diff(seq.ages) >= 0).all()
        assert (np.diff(seq.seasons if hasattr(seq, "seasons") else seq.ages) >= 0).all()


def test_build_athlete_sequence_feature_values(summary):
    seqs = build_athlete_sequences(summary)
    seq = next(iter(seqs.values()))
    assert np.isfinite(seq.features).all()
    assert (seq.features[:, FEATURE_COLUMNS.index("n_races")] >= 1).all()
    assert seq.features[:, FEATURE_COLUMNS.index("best_level")].min() >= 1
    assert seq.features[:, FEATURE_COLUMNS.index("best_level")].max() <= 5


def test_build_training_samples_shapes(seqs):
    sample_seqs = dict(list(seqs.items())[:20])
    data = build_training_samples(sample_seqs, history_len=10, horizons=5, min_history=3, fit_scalers=True)
    N, L, F = data.X.shape
    assert L == 10
    assert F == len(FEATURE_COLUMNS)
    assert data.y.shape == (N, 5)
    assert data.y_mask.shape == (N, 5)
    assert (data.lengths <= 10).all()
    assert (data.lengths >= 3).all()
    # every sample has at least one valid target
    assert (data.y_mask.sum(axis=1) >= 1).all()


def test_build_training_samples_right_aligned(seqs):
    sample_seqs = dict(list(seqs.items())[:5])
    data = build_training_samples(sample_seqs, history_len=10, horizons=5, min_history=3, fit_scalers=True)
    # last valid position always holds the most recent season
    scaler = data.scaler_X
    for i, length in enumerate(data.lengths):
        last = data.X[i, -1]
        assert not np.allclose(last, 0.0) or length == 0


def test_scalers_roundtrip(seqs):
    sample_seqs = dict(list(seqs.items())[:15])
    data = build_training_samples(sample_seqs, history_len=10, horizons=5, min_history=3, fit_scalers=True)
    orig = np.array([[11.0], [11.5], [12.0], [12.5]])
    scaled = data.scaler_y.transform(orig)
    back = data.scaler_y.inverse_transform(scaled)
    assert np.allclose(orig, back, atol=1e-6)
    # feature scaler roundtrip
    f_orig = np.array([[18.0, 11.4, 8.0, 3.0, 3.0, 0.2, 5.0]])
    f_back = data.scaler_X.inverse_transform(data.scaler_X.transform(f_orig))
    assert np.allclose(f_orig, f_back, atol=1e-6)


def test_build_training_samples_deterministic(seqs):
    sample_seqs = dict(list(seqs.items())[:10])
    d1 = build_training_samples(sample_seqs, 10, 5, 3, fit_scalers=True)
    d2 = build_training_samples(sample_seqs, 10, 5, 3, fit_scalers=True)
    assert np.array_equal(d1.X, d2.X)
    assert np.array_equal(d1.y, d2.y)
