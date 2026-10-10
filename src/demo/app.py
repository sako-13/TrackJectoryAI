"""Live demo: forecast your own career progression.

Serves a static web app plus a JSON API:

    GET  /health    -> liveness probe
    GET  /meta      -> model/event metadata
    GET  /          -> demo web page
    POST /predict   -> {event, seasons, models} -> forecasts + archetype

The models are the artifacts produced by `python scripts/run_pipeline.py`
(artifacts/model_bundle.pkl).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.clustering.archetypes import TRAJECTORY_FEATURES, _athlete_trajectory_row
from src.data.features import FEATURE_COLUMNS
from src.models.base import History

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_PATH = REPO_ROOT / "artifacts" / "model_bundle.pkl"
STATIC_DIR = Path(__file__).parent / "static"

BUNDLE: dict | None = None


def get_bundle() -> dict:
    global BUNDLE
    if BUNDLE is None:
        if not BUNDLE_PATH.exists():
            raise RuntimeError(
                "Model artifacts not found. Run the pipeline first: "
                "python scripts/run_pipeline.py"
            )
        with open(BUNDLE_PATH, "rb") as f:
            BUNDLE = pickle.load(f)
    return BUNDLE


app = FastAPI(title="Athlete Career Forecaster", version="0.1.0")


class SeasonInput(BaseModel):
    age: float
    season_best: float
    n_races: int = 5
    level: int = 3

    @field_validator("age")
    @classmethod
    def check_age(cls, v):
        if not 10 <= v <= 50:
            raise ValueError("age must be between 10 and 50")
        return v

    @field_validator("level")
    @classmethod
    def check_level(cls, v):
        if v not in (1, 2, 3, 4, 5):
            raise ValueError("level must be 1..5")
        return v


class PredictRequest(BaseModel):
    event: str = "100m"
    seasons: list[SeasonInput]
    models: list[str] | None = None

    @field_validator("seasons")
    @classmethod
    def check_seasons(cls, v):
        if len(v) < 3:
            raise ValueError("provide at least 3 seasons of progression")
        return sorted(v, key=lambda s: s.age)


def _build_history(request: PredictRequest, feature_columns: list[str]) -> History:
    rows = [s.model_dump() for s in request.seasons]
    df = pd.DataFrame(rows).sort_values("age").reset_index(drop=True)
    df["best_level"] = df["level"]
    df["mean_level"] = df["level"]
    df["delta_best"] = (df["season_best"].shift(1) - df["season_best"]).fillna(0.0)
    df["seasons_so_far"] = np.arange(1, len(df) + 1)
    feats = df[feature_columns].to_numpy(dtype=float)
    return History(
        athlete_id="demo",
        ages=df["age"].to_numpy(dtype=float),
        season_best=df["season_best"].to_numpy(dtype=float),
        n_races=df["n_races"].to_numpy(dtype=float),
        levels=df["best_level"].to_numpy(dtype=float),
        features=feats,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts": BUNDLE_PATH.exists()}


@app.get("/meta")
def meta() -> dict:
    bundle = get_bundle()
    return {
        "event": bundle["event"],
        "quantiles": bundle["quantiles"],
        "horizons": bundle["horizons"],
        "models": list(bundle["models"].keys()),
        "archetypes": sorted(set(bundle["cluster"]["archetypes"].values())),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    bundle = get_bundle()
    if request.event != bundle["event"]:
        raise HTTPException(
            status_code=400,
            detail=f"This model bundle was trained for {bundle['event']}. "
                   f"Re-run the pipeline with --event {request.event}.",
        )
    feature_columns = bundle["feature_columns"]
    history = _build_history(request, feature_columns)
    horizons = bundle["horizons"]
    quantiles = bundle["quantiles"]

    model_names = request.models or list(bundle["models"].keys())
    forecasts = {}
    for name in model_names:
        if name not in bundle["models"]:
            raise HTTPException(status_code=400, detail=f"unknown model {name!r}")
        fc = bundle["models"][name].forecast(history, horizons)
        rows = [
            {
                "horizon": h + 1,
                "age": round(float(history.ages[-1]) + h + 1, 1),
                **{
                    f"q{round(t * 100)}": round(float(fc[h, j]), 3)
                    for j, t in enumerate(quantiles)
                },
            }
            for h in range(min(horizons, fc.shape[0]))
        ]
        # apply the same conformal recalibration used in evaluation
        conformal = bundle.get("conformal", {}).get(name)
        if conformal is not None:
            rows = conformal.apply_forecast(rows)
            for row in rows:
                row["q10"] = round(max(row["q10"], bundle["bounds"][0]), 3)
                row["q90"] = round(min(row["q90"], bundle["bounds"][1]), 3)
        forecasts[name] = rows

    # archetype guess from the runner's progression so far
    n_seasons = len(history.ages)
    row = _athlete_trajectory_row(
        pd.DataFrame(
            {
                "athlete_id": ["demo"] * n_seasons,
                "season": list(range(2020, 2020 + n_seasons)),
                "age": history.ages.tolist(),
                "season_best": history.season_best.tolist(),
            }
        )
    )
    feature_row = pd.DataFrame([{k: row[k] for k in TRAJECTORY_FEATURES}])
    cluster = bundle["cluster"]
    X_scaled = cluster["scaler"].transform(feature_row[cluster["feature_names"]].to_numpy(float))
    cid = int(cluster["kmeans"].predict(X_scaled)[0])

    return {
        "event": request.event,
        "seasons": [s.model_dump() for s in request.seasons],
        "forecasts": forecasts,
        "archetype": cluster["archetypes"][cid],
        "archetype_cluster": cid,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
