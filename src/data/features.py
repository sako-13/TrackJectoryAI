"""Feature engineering: per-athlete sequences and training samples."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Order matters: this is the feature vector fed to the sequence model.
FEATURE_COLUMNS = [
    "age",
    "season_best",
    "n_races",
    "best_level",
    "mean_level",
    "delta_best",
    "seasons_so_far",
]


@dataclass
class AthleteSequence:
    """All observed seasons of one athlete, in chronological order."""

    athlete_id: int
    ages: np.ndarray            # (T,)
    season_best: np.ndarray     # (T,) seconds
    n_races: np.ndarray         # (T,)
    levels: np.ndarray          # (T,) best competition level per season
    features: np.ndarray        # (T, F) raw engineered features
    sex: str | None = None

    @property
    def T(self) -> int:
        return len(self.season_best)


@dataclass
class TrainingData:
    """Padded sequences for the sequence model."""

    X: np.ndarray               # (N, L, F) front-padded (right-aligned history)
    lengths: np.ndarray         # (N,) true history lengths
    y: np.ndarray               # (N, H) future season bests (0 where masked)
    y_mask: np.ndarray          # (N, H) 1.0 where the target exists
    athlete_ids: np.ndarray     # (N,)
    cutoffs: np.ndarray         # (N,) number of observed seasons at each sample
    scaler_X: StandardScaler
    scaler_y: StandardScaler


def build_athlete_sequences(
    summary: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict[int, AthleteSequence]:
    """Convert the season summary into per-athlete chronological sequences."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    seqs: dict[int, AthleteSequence] = {}
    for athlete_id, g in summary.groupby("athlete_id", sort=True):
        g = g.sort_values("season")
        feats = g[feature_columns].copy()
        feats["delta_best"] = feats["delta_best"].fillna(0.0)
        seqs[int(athlete_id)] = AthleteSequence(
            athlete_id=int(athlete_id),
            ages=g["age"].to_numpy(dtype=float),
            season_best=g["season_best"].to_numpy(dtype=float),
            n_races=g["n_races"].to_numpy(dtype=float),
            levels=g["best_level"].to_numpy(dtype=float),
            features=feats.to_numpy(dtype=float),
            sex=g["sex"].iloc[0] if "sex" in g.columns else None,
        )
    return seqs


def _pad_front(sequences: list[np.ndarray], history_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-align sequences: real data at the end, zeros in front."""
    n = len(sequences)
    width = sequences[0].shape[1] if sequences[0].ndim == 2 else 1
    if sequences[0].ndim == 2:
        padded = np.zeros((n, history_len, width), dtype=np.float32)
    else:
        padded = np.zeros((n, history_len), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        lengths[i] = length
        if sequences[0].ndim == 2:
            padded[i, history_len - length :] = seq
        else:
            padded[i, history_len - length :] = seq
    return padded, lengths


def build_training_samples(
    sequences: dict[int, AthleteSequence],
    history_len: int,
    horizons: int,
    min_history: int,
    fit_scalers: bool = False,
    scaler_X: StandardScaler | None = None,
    scaler_y: StandardScaler | None = None,
) -> TrainingData:
    """Build sliding-window training samples.

    For each athlete and each cutoff ``c`` (number of observed seasons, from
    ``min_history`` to ``T-1``), the sample's history is the last
    ``history_len`` seasons before ``c`` and the targets are the next
    ``horizons`` seasons (masked when the career ends).
    """
    hist_list, y_list, mask_list, ids, cutoffs = [], [], [], [], []
    for aid, seq in sequences.items():
        T = seq.T
        for c in range(min_history, T):
            hist_list.append(seq.features[max(0, c - history_len) : c])
            targets = seq.season_best[c : c + horizons]
            y = np.zeros(horizons, dtype=np.float64)
            m = np.zeros(horizons, dtype=np.float64)
            y[: len(targets)] = targets
            m[: len(targets)] = 1.0
            y_list.append(y)
            mask_list.append(m)
            ids.append(aid)
            cutoffs.append(c)

    X, lengths = _pad_front(hist_list, history_len)
    y = np.stack(y_list)
    y_mask = np.stack(mask_list)

    valid = np.zeros((X.shape[0], history_len), dtype=bool)
    for i, length in enumerate(lengths):
        valid[i, history_len - length :] = True

    if fit_scalers or scaler_X is None:
        scaler_X = StandardScaler().fit(X[valid])
    if fit_scalers or scaler_y is None:
        scaler_y = StandardScaler().fit(y[y_mask > 0].reshape(-1, 1))

    X_scaled = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)

    return TrainingData(
        X=X_scaled,
        lengths=lengths,
        y=y_scaled,
        y_mask=y_mask,
        athlete_ids=np.array(ids, dtype=int),
        cutoffs=np.array(cutoffs, dtype=int),
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )


def build_history(athlete_id: int, seq: AthleteSequence, cutoff: int, history_len: int | None = None) -> "History":
    """Build a :class:`~src.models.base.History` for forecasting at a cutoff.

    When ``history_len`` is given, only the last ``history_len`` observed
    seasons are kept (matching the sequence-model training windows).
    """
    from src.models.base import History

    c = min(cutoff, seq.T)
    start = max(0, c - history_len) if history_len else 0
    return History(
        athlete_id=athlete_id,
        ages=seq.ages[start:c],
        season_best=seq.season_best[start:c],
        n_races=seq.n_races[start:c],
        levels=seq.levels[start:c],
        features=seq.features[start:c],
    )
