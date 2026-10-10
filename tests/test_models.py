"""Tests for the forecasters (baselines + sequence models)."""
from __future__ import annotations

import numpy as np
import torch

from src.models.base import History
from src.models.sequence import SeqForecaster, quantile_loss


def _check_quantile_forecast(fc, horizons, n_quantiles=3):
    assert fc.shape == (horizons, n_quantiles)
    assert np.isfinite(fc).all()
    # quantiles must be ordered
    assert (np.diff(fc, axis=1) >= -1e-9).all()


def test_age_curve_forecast(trained, sample_history):
    fc = trained["age_curve"].forecast(sample_history, 5)
    _check_quantile_forecast(fc, 5)
    assert fc.min() >= 9.4 and fc.max() <= 16.0


def test_age_curve_young_athlete_improves(trained):
    # a 16-19 year old still on the steep part of the age curve
    hist = History(
        athlete_id=1,
        ages=np.array([16.0, 17.0, 18.0, 19.0]),
        season_best=np.array([12.5, 12.2, 11.9, 11.7]),
        n_races=np.array([4, 5, 6, 7]),
        levels=np.array([4, 4, 3, 3]),
        features=np.column_stack([
            [16.0, 17.0, 18.0, 19.0], [12.5, 12.2, 11.9, 11.7],
            [4, 5, 6, 7], [4, 4, 3, 3], [4, 4, 3, 3],
            [0, 0.3, 0.3, 0.2], [1, 2, 3, 4],
        ]),
    )
    fc = trained["age_curve"].forecast(hist, 3)
    assert fc[2, 1] < hist.season_best[-1]  # median improves by horizon 3


def test_gbm_forecast(trained, sample_history):
    fc = trained["gbm"].forecast(sample_history, 5)
    _check_quantile_forecast(fc, 5)


def test_lstm_forecast(trained, sample_history):
    fc = trained["lstm"].forecast(sample_history, 5)
    _check_quantile_forecast(fc, 5)


def test_lstm_forecast_short_history(trained, sample_history):
    hist = History(
        athlete_id=sample_history.athlete_id,
        ages=sample_history.ages[:3],
        season_best=sample_history.season_best[:3],
        n_races=sample_history.n_races[:3],
        levels=sample_history.levels[:3],
        features=sample_history.features[:3],
    )
    fc = trained["lstm"].forecast(hist, 5)
    _check_quantile_forecast(fc, 5)


def test_quantile_loss_value():
    pred = torch.tensor([[[0.0]]])      # (N=1, H=1, Q=1)
    target = torch.tensor([[1.0]])      # (N=1, H=1)
    mask = torch.tensor([[1.0]])
    loss = quantile_loss(pred, target, mask, [0.5])
    assert abs(loss.item() - 0.5) < 1e-6  # pinball at tau=0.5, error=1


def test_seq_forecaster_forward_shapes():
    for kind in ("lstm", "transformer"):
        model = SeqForecaster(n_features=7, hidden=16, horizons=5,
                              quantiles=(0.1, 0.5, 0.9), kind=kind,
                              history_len=10, num_layers=1)
        x = torch.randn(4, 10, 7)
        lengths = torch.tensor([10, 7, 4, 3])
        out = model(x, lengths)
        assert out.shape == (4, 5, 3)


def test_lstm_can_overfit(splits, config):
    """A sequence model must memorize a single athlete (capacity check)."""
    from src.data.features import build_training_samples
    from src.models.sequence import SequenceModelAdapter

    train, _, _ = splits
    single = dict(list(train.items())[:1])
    adapter = SequenceModelAdapter(
        kind="lstm", quantiles=(0.1, 0.5, 0.9), epochs=150,
        batch_size=16, lr=5e-3, patience=150,
    )
    adapter.fit(single, None, config)
    data = build_training_samples(
        single, adapter.history_len, adapter.horizons, adapter.min_history,
        scaler_X=adapter.scaler_X, scaler_y=adapter.scaler_y,
    )
    with torch.no_grad():
        pred = adapter.model(
            torch.tensor(data.X, dtype=torch.float32),
            torch.tensor(data.lengths, dtype=torch.long),
        )
        loss = quantile_loss(
            pred, torch.tensor(data.y), torch.tensor(data.y_mask),
            adapter.quantiles,
        )
    assert loss.item() < 0.08
