"""Shared fixtures: small synthetic athlete population + trained models."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.data.clean import build_season_summary, clean_results
from src.data.features import build_athlete_sequences
from src.data.ingest import generate_synthetic_dataset
from src.models.age_curve import AgeCurveForecaster
from src.models.gbm import GBMForecaster
from src.models.sequence import SequenceModelAdapter

N_ATHLETES = 60


@pytest.fixture(scope="session")
def config():
    return load_config(
        overrides={
            "data": {"n_synthetic_athletes": N_ATHLETES, "min_seasons": 4},
            "model": {"epochs": 2, "batch_size": 32, "early_stop_patience": 2},
            "clustering": {"k": 4},
        }
    )


@pytest.fixture(scope="session")
def raw_df(config):
    return generate_synthetic_dataset(config["event"], N_ATHLETES, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df, config):
    return clean_results(raw_df, config["event"])


@pytest.fixture(scope="session")
def summary(clean_df, config):
    return build_season_summary(clean_df, min_seasons=config["data"]["min_seasons"])


@pytest.fixture(scope="session")
def seqs(summary):
    return build_athlete_sequences(summary)


@pytest.fixture(scope="session")
def splits(seqs, config):
    ids = sorted(seqs.keys())
    rng = np.random.default_rng(config["seed"])
    perm = rng.permutation(len(ids))
    n_train = int(len(perm) * config["split"]["train"])
    n_val = int(len(perm) * config["split"]["val"])
    train_ids = set(np.array(ids)[perm[:n_train]])
    val_ids = set(np.array(ids)[perm[n_train : n_train + n_val]])
    test_ids = set(np.array(ids)[perm[n_train + n_val :]])
    return (
        {k: v for k, v in seqs.items() if k in train_ids},
        {k: v for k, v in seqs.items() if k in val_ids},
        {k: v for k, v in seqs.items() if k in test_ids},
    )


@pytest.fixture(scope="session")
def trained(splits, config):
    train, val, _ = splits
    quantiles = tuple(config["model"]["quantiles"])
    bounds = (9.4, 16.0) if config["event"] == "100m" else (99.0, 170.0)
    models = {
        "age_curve": AgeCurveForecaster(quantiles=quantiles, bounds=bounds).fit(train, val, config),
        "gbm": GBMForecaster(quantiles=quantiles, horizons=5, random_state=42, bounds=bounds).fit(train, val, config),
        "lstm": SequenceModelAdapter(
            kind="lstm", quantiles=quantiles, epochs=2, batch_size=32,
            patience=2, bounds=bounds,
        ).fit(train, val, config),
    }
    return models


@pytest.fixture()
def sample_history(trained, splits):
    """A young athlete's observed history for forecast tests."""
    _, _, test = splits
    aid = sorted(test.keys())[0]
    seq = test[aid]
    from src.models.base import History

    return History(
        athlete_id=aid,
        ages=seq.ages[:4],
        season_best=seq.season_best[:4],
        n_races=seq.n_races[:4],
        levels=seq.levels[:4],
        features=seq.features[:4],
    )
