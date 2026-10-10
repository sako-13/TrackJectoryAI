"""End-to-end forecasting pipeline.

Stages
------
1. Ingest race results (World Athletics API when reachable, realistic
   synthetic data otherwise, on-disk cache in between)
2. Clean and validate the raw records
3. Aggregate to athlete-seasons and engineer features
4. Grouped split of athletes (train / validation / test)
5. Train: age-curve baseline, gradient-boosted baseline, LSTM/Transformer
   sequence model with quantile outputs
6. Evaluate on never-seen test athletes at every career cutoff
7. Check prediction-interval calibration
8. Cluster career trajectories into development archetypes
9. Produce figures, metrics and the written analysis
10. Persist artifacts for the demo app
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.archetypes import (
    cluster_athletes,
    describe_clusters,
    predict_archetype,
    trajectory_features,
)
from src.config import load_config
from src.data.clean import build_season_summary, clean_results
from src.data.features import (
    FEATURE_COLUMNS,
    AthleteSequence,
    build_athlete_sequences,
    build_history,
)
from src.data.ingest import EVENT_BOUNDS, fetch_event_dataset
from src.evaluation.calibration import calibration_report
from src.evaluation.conformal import ConformalScaler
from src.evaluation.metrics import summarize_records
from src.models.age_curve import AgeCurveForecaster
from src.models.gbm import GBMForecaster
from src.models.sequence import SequenceModelAdapter
from src.utils import ensure_dir, save_json, set_seed
from src import viz

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """Run the full pipeline and return a summary dictionary."""
    cfg = load_config(config_path, overrides)
    root = Path(project_root).resolve() if project_root else REPO_ROOT
    event = cfg["event"]
    seed = cfg["seed"]
    quantiles = tuple(cfg["model"]["quantiles"])
    horizons = cfg["features"]["horizons"]
    history_len = cfg["features"]["history_len"]
    min_history = cfg["features"]["min_history"]
    bounds = EVENT_BOUNDS[event]
    set_seed(seed)

    ensure_dir(root / cfg["data"]["processed_dir"])
    ensure_dir(root / cfg["reports_dir"])
    ensure_dir(root / cfg["artifacts_dir"])

    print(f"[1/10] Ingesting {event} data ...")
    raw, source = fetch_event_dataset(
        event=event,
        n_athletes=cfg["data"]["n_synthetic_athletes"],
        seed=seed,
        cache_dir=root / cfg["data"]["cache_dir"],
        source=cfg["data"]["source"],
    )
    print(f"      source={source}, rows={len(raw):,}")

    print("[2/10] Cleaning race records ...")
    clean = clean_results(raw, event)
    clean.to_csv(root / cfg["data"]["processed_dir"] / f"clean_{event}.csv", index=False)

    print("[3/10] Building season summaries and features ...")
    summary = build_season_summary(clean, min_seasons=cfg["data"]["min_seasons"])
    summary.to_csv(root / cfg["data"]["processed_dir"] / f"season_summary_{event}.csv", index=False)
    seqs = build_athlete_sequences(summary)
    n_athletes = len(seqs)
    n_seasons = int(len(summary))
    print(f"      athletes={n_athletes:,} athlete-seasons={n_seasons:,}")

    # ---- grouped split: no athlete appears in two splits ----
    athlete_ids = sorted(seqs.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(athlete_ids))
    n_train = int(len(perm) * cfg["split"]["train"])
    n_val = int(len(perm) * cfg["split"]["val"])
    train_ids = set(np.array(athlete_ids)[perm[:n_train]])
    val_ids = set(np.array(athlete_ids)[perm[n_train : n_train + n_val]])
    test_ids = set(np.array(athlete_ids)[perm[n_train + n_val :]])
    train_seqs = {k: v for k, v in seqs.items() if k in train_ids}
    val_seqs = {k: v for k, v in seqs.items() if k in val_ids}
    test_seqs = {k: v for k, v in seqs.items() if k in test_ids}
    print(f"      split: train={len(train_seqs)} val={len(val_seqs)} test={len(test_seqs)}")

    print("[4/10] Training models ...")
    age_curve = AgeCurveForecaster(quantiles=quantiles, bounds=bounds).fit(train_seqs, val_seqs, cfg)
    gbm = GBMForecaster(quantiles=quantiles, horizons=horizons, random_state=seed, bounds=bounds).fit(
        train_seqs, val_seqs, cfg
    )
    seq_model = SequenceModelAdapter(
        kind=cfg["model"]["kind"],
        quantiles=quantiles,
        history_len=history_len,
        horizons=horizons,
        min_history=min_history,
        hidden=cfg["model"]["hidden"],
        num_layers=cfg["model"]["num_layers"],
        nhead=cfg["model"]["nhead"],
        ff_dim=cfg["model"]["ff_dim"],
        dropout=cfg["model"]["dropout"],
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        lr=cfg["model"]["lr"],
        patience=cfg["model"]["early_stop_patience"],
        seed=seed,
        bounds=bounds,
    ).fit(train_seqs, val_seqs, cfg)
    models = {"age_curve": age_curve, "gbm": gbm, cfg["model"]["kind"]: seq_model}

    def _evaluate(model, athlete_ids) -> list[dict]:
        """Collect forecast records for a set of athletes at every cutoff."""
        records = []
        for aid in sorted(athlete_ids):
            seq = seqs[aid]
            T = seq.T
            for cutoff in range(min_history, T):
                h_max = min(horizons, T - cutoff)
                history = build_history(aid, seq, cutoff)
                fc = model.forecast(history, h_max)
                for h in range(h_max):
                    rec = {
                        "athlete_id": aid,
                        "horizon": h + 1,
                        "cutoff": float(seq.ages[cutoff - 1]),
                        "age_at_forecast": float(seq.ages[cutoff + h]),
                        "y": float(seq.season_best[cutoff + h]),
                    }
                    for j, tau in enumerate(quantiles):
                        rec[f"q{round(tau * 100)}"] = float(fc[h, j])
                    records.append(rec)
        return records

    print("[5/10] Evaluating on validation and held-out test athletes ...")
    records_by_model: dict[str, list[dict]] = {}
    conformal_by_model: dict[str, ConformalScaler] = {}
    for name, model in models.items():
        # calibrate interval widths on validation athletes, then
        # report performance on the never-seen test athletes
        val_records = _evaluate(model, val_ids)
        test_records = _evaluate(model, test_ids)
        conformal = ConformalScaler(target=0.80).fit(val_records)
        conformal_by_model[name] = conformal
        records_by_model[name] = conformal.apply(test_records)

    metrics = {name: summarize_records(recs, quantiles) for name, recs in records_by_model.items()}
    print("[6/10] Checking interval calibration ...")
    calibration = {name: calibration_report(recs, quantiles) for name, recs in records_by_model.items()}

    print("[7/10] Clustering career trajectories ...")
    traj = trajectory_features(summary)
    cluster_result = cluster_athletes(traj, k=cfg["clustering"]["k"], seed=seed)
    cluster_desc = describe_clusters(cluster_result)
    traj.to_csv(root / cfg["reports_dir"] / "trajectory_features.csv")
    cluster_desc.to_csv(root / cfg["reports_dir"] / "archetypes.csv", index=False)

    print("[8/10] Producing figures ...")
    fig_dir = ensure_dir(root / cfg["reports_dir"] / "figures")
    figs = {
        "career_trajectories": viz.plot_career_trajectories(summary, fig_dir / "career_trajectories.png", event),
        "forecast_examples": viz.plot_forecast_examples(records_by_model, summary, fig_dir / "forecast_examples.png", event),
        "metrics_by_horizon": viz.plot_metrics_by_horizon(metrics, fig_dir / "metrics_by_horizon.png"),
        "model_comparison": viz.plot_model_comparison(metrics, fig_dir / "model_comparison.png"),
        "calibration": viz.plot_calibration(calibration, fig_dir / "calibration.png"),
        "cluster_scatter": viz.plot_cluster_scatter(cluster_result, fig_dir / "cluster_scatter.png"),
        "cluster_trajectories": viz.plot_cluster_trajectories(cluster_result, summary, fig_dir / "cluster_trajectories.png", event),
    }
    best_model = min(metrics, key=lambda m: metrics[m]["overall"]["mae"])
    figs["residuals_by_age"] = viz.plot_residuals_by_age(
        records_by_model[best_model], fig_dir / "residuals_by_age.png"
    )

    print("[9/10] Saving artifacts ...")
    artifacts_dir = ensure_dir(root / cfg["artifacts_dir"])
    bundle = {
        "event": event,
        "quantiles": list(quantiles),
        "horizons": horizons,
        "history_len": history_len,
        "feature_columns": FEATURE_COLUMNS,
        "bounds": list(bounds),
        "models": models,
        "conformal": conformal_by_model,
        "cluster": {
            "kmeans": cluster_result["kmeans"],
            "scaler": cluster_result["scaler"],
            "pca": cluster_result["pca"],
            "archetypes": cluster_result["archetypes"],
            "feature_names": cluster_result["feature_names"],
        },
        "config": cfg,
    }
    with open(artifacts_dir / "model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    save_json(
        {
            "event": event,
            "data_source": source,
            "n_athletes": n_athletes,
            "n_athlete_seasons": n_seasons,
            "split": {"train": len(train_seqs), "val": len(val_seqs), "test": len(test_seqs)},
            "metrics": metrics,
            "calibration": calibration,
            "best_model": best_model,
            "archetypes": cluster_desc.to_dict("records"),
        },
        root / cfg["reports_dir"] / "metrics.json",
    )

    print("[10/10] Writing analysis report ...")
    analysis = _render_analysis(event, source, n_athletes, n_seasons, metrics, calibration, cluster_desc, best_model)
    report_path = root / cfg["reports_dir"] / "analysis.md"
    report_path.write_text(analysis)

    print(f"Done. Best model: {best_model}. Report: {report_path}")
    return {
        "event": event,
        "source": source,
        "n_athletes": n_athletes,
        "metrics": metrics,
        "calibration": calibration,
        "best_model": best_model,
        "archetypes": cluster_desc,
        "figures": {k: str(v) for k, v in figs.items()},
    }


def _render_analysis(event, source, n_athletes, n_seasons, metrics, calibration, cluster_desc, best_model) -> str:
    """Render the written analysis report (Markdown)."""
    lines = []
    lines.append(f"# Career-Forecast Analysis — {event}")
    lines.append("")
    lines.append(f"**Data source:** {source}  ")
    lines.append(f"**Athletes evaluated:** {n_athletes:,}  ")
    lines.append(f"**Athlete-seasons:** {n_seasons:,}  ")
    lines.append("**Evaluation:** every test athlete was forecast from every career "
                 "cutoff (≥3 observed seasons); horizons 1–5 seasons; metrics "
                 "aggregated per horizon. The split is at the *athlete* level, so "
                 "no test athlete was seen during training.")
    lines.append("")
    lines.append("## 1. Overall accuracy")
    lines.append("")
    lines.append("| Model | MAE (s) | RMSE (s) | CRPS (s) | MAPE (%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in metrics.items():
        o = m["overall"]
        lines.append(f"| {name} | {o['mae']:.3f} | {o['rmse']:.3f} | {o['crps']:.3f} | {o['mape']:.2f} |")
    lines.append("")
    lines.append(f"**Best model: {best_model}.**")
    lines.append("")
    lines.append("### Accuracy by horizon (best model)")
    lines.append("")
    lines.append("| Horizon | MAE (s) | RMSE (s) | CRPS (s) |")
    lines.append("|---:|---:|---:|---:|")
    for h in metrics[best_model]["per_horizon"]:
        lines.append(f"| {h['horizon']} | {h['mae']:.3f} | {h['rmse']:.3f} | {h['crps']:.3f} |")
    lines.append("")
    lines.append("## 2. Prediction-interval calibration")
    lines.append("")
    lines.append("Each model emits 10%/50%/90% quantiles, i.e. an 80% prediction "
                 "interval. Interval widths are recalibrated with a per-horizon "
                 "conformal scale factor fitted on the *validation* athletes, so "
                 "the reported coverage on test athletes is honest. A calibrated "
                 "model covers the true season best 80% of the time (within the "
                 "Wilson 95% confidence band).")
    lines.append("")
    lines.append("| Model | Empirical coverage | Nominal | 95% CI | Mean width (s) | Calibrated? |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, cal in calibration.items():
        o = cal["overall"]
        lines.append(
            f"| {name} | {o['empirical_coverage']:.3f} | {o['nominal']:.2f} | "
            f"[{o['coverage_ci'][0]:.3f}, {o['coverage_ci'][1]:.3f}] | {o['mpiw']:.3f} | "
            f"{'yes' if o['calibrated'] else 'no'} |"
        )
    lines.append("")
    lines.append("### Coverage by horizon")
    lines.append("")
    lines.append("| Model | H=1 | H=2 | H=3 | H=4 | H=5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, cal in calibration.items():
        cells = " | ".join(f"{h['picp']:.2f}" for h in cal["per_horizon"])
        lines.append(f"| {name} | {cells} |")
    lines.append("")
    lines.append("## 3. Development archetypes")
    lines.append("")
    lines.append("Career trajectories (peak age, improvement rate, career length, "
                 "gains) were standardized and clustered with K-Means "
                 f"(k={len(cluster_desc)}). Archetypes are labelled from the "
                 "cluster centers: low peak age → *Early Peaker*, high peak age → "
                 "*Late Bloomer*, high improvement rate → *Rapid Riser*, else "
                 "*Steady Developer*.")
    lines.append("")
    lines.append("| Archetype | Athletes | Peak age | Peak best (s) | Total gain (s) | Improvement rate (s/yr) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in cluster_desc.to_dict("records"):
        lines.append(
            f"| {row['archetype']} | {row['n_athletes']} | {row['mean_peak_age']:.1f} | "
            f"{row['mean_peak_best']:.2f} | {row['mean_total_gain']:.2f} | {row['mean_improvement_rate']:.2f} |"
        )
    lines.append("")
    lines.append("## 4. Limitations")
    lines.append("")
    lines.append("- **Data provenance:** World Athletics does not publish an official "
                 "public API. The ingestion client targets the site's unofficial "
                 "endpoints; when those are unreachable the pipeline trains on a "
                 "realistic *synthetic* population with the same schema, so all "
                 "reported numbers reflect the simulated population unless a live "
                 "cache from the API is present in `data/raw/`.")
    lines.append("- **Single-event scope:** models are trained per event; cross-event "
                 "transfer is not attempted.")
    lines.append("- **Career censoring:** athletes who retire early are absent from the "
                 "data, which biases long-horizon estimates upward for continuing "
                 "athletes.")
    lines.append("- **Quantile resolution:** intervals are built from only three "
                 "quantile levels; richer distributional heads would sharpen CRPS.")
    lines.append("- **No contextual covariates:** training load, injuries, coaching "
                 "changes and altitude are not available.")
    lines.append("")
    return "\n".join(lines)
