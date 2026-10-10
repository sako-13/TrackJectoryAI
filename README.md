# Athlete Career Forecaster

Probabilistic forecasting of track-athlete career development. Trains an
LSTM/Transformer sequence model on season-by-season results for thousands of
sprinters, predicts each athlete's next 5 seasons with uncertainty ranges,
benchmarks it against an age-curve regression and a gradient-boosted model,
and discovers career-development archetypes (early peakers, late bloomers,
rapid risers, steady developers) via unsupervised clustering. Ships a live
web demo where a runner can enter their own progression and get a projected
range.

## Results (100m, 2,228 athletes / 22,004 athlete-seasons)

Evaluation on **never-seen athletes** (athlete-level split), forecasting
every career from every cutoff (≥ 3 observed seasons), horizons 1–5:

| Model | MAE (s) | RMSE (s) | CRPS (s) | 80% interval coverage | Mean width (s) |
|---|---:|---:|---:|---:|---:|
| Age-curve regression | 0.150 | 0.204 | 0.127 | 0.84 | 0.53 |
| Gradient boosting | 0.103 | 0.147 | 0.089 | 0.84 | 0.37 |
| **LSTM (quantile)** | **0.090** | **0.133** | **0.078** | 0.82 | 0.32 |

The sequence model is most accurate at every horizon (MAE 0.049 s at H=1 →
0.152 s at H=5) and produces the *sharpest* calibrated intervals. Interval
widths are recalibrated with a per-horizon conformal scale fitted on
validation athletes; coverage is checked against the nominal 80% level with
athlete-level Wilson confidence intervals (all three models calibrated).

**Development archetypes** (K-Means on trajectory-shape features):

| Archetype | Athletes | Mean peak age | Mean total gain (s) |
|---|---:|---:|---:|
| Early Peaker | 590 | 23.4 | 0.47 |
| Late Bloomer | 510 | 25.1 | 0.94 |
| Rapid Riser | 533 | 23.7 | 0.99 |
| Steady Developer | 595 | 24.1 | 0.49 |

## Quickstart

```bash
pip3 install -r requirements.txt

# run the full pipeline (ingest → clean → features → train → evaluate →
# calibrate → cluster → figures → report → demo artifacts)
python3 scripts/run_pipeline.py            # full run (~2–4 min on CPU)
python3 scripts/run_pipeline.py --quick    # smoke run

# run the test suite (62 tests)
python3 -m pytest

# launch the live demo
python3 scripts/serve_demo.py              # → http://127.0.0.1:8000
```

Or use the Makefile: `make install`, `make test`, `make pipeline`, `make demo`.

## How it works

```
src/
├── data/
│   ├── ingest.py       # World Athletics client + realistic synthetic generator
│   ├── clean.py        # validation, wind-aided marks, season aggregation
│   └── features.py     # per-athlete sequences, sliding-window training samples
├── models/
│   ├── age_curve.py    # baseline 1: quadratic age curve + athlete offset
│   ├── gbm.py          # baseline 2: recursive multi-horizon gradient boosting
│   └── sequence.py     # LSTM / Transformer with quantile-regression head
├── evaluation/
│   ├── metrics.py      # MAE/RMSE/MAPE, pinball loss, CRPS, PICP/MPIW
│   ├── calibration.py  # coverage vs. nominal, Wilson CIs
│   └── conformal.py    # per-horizon conformal interval recalibration
├── clustering/
│   └── archetypes.py   # trajectory features, K-Means, archetype labelling
├── demo/               # FastAPI backend + vanilla-JS web app
├── viz.py              # all report figures
└── pipeline.py         # end-to-end orchestration
```

### Data

World Athletics does not publish an official public API. `WorldAthleticsClient`
targets the unofficial JSON endpoints that worldathletics.org itself uses
(profile, results, search) with on-disk caching under `data/raw/`. When the
API is unreachable — as it is behind anti-bot protection at the time of
writing — the pipeline falls back to `generate_synthetic_dataset`, which
simulates a realistic athlete population with the same schema: latent talent
tiers, a peak-age mixture (early / mid / late), first-season level, total
career improvement, post-peak decline, missed seasons, meet-level
hierarchies, and wind readings. Everything downstream is identical either way;
drop a live `data/raw/results_100m.csv` cache into the repo to train on real
data. Small committed samples live in `data/sample/`.

### Models

- **Age-curve baseline** — quadratic performance-vs-age curve fitted on all
  training athletes, combined with a mildly mean-reverting athlete offset;
  per-horizon residual-quantile intervals from validation athletes.
- **Gradient-boosted baseline** — `HistGradientBoostingRegressor` on career
  context (age, season best, races, level, deltas, rolling stats), rolled
  out recursively for horizons 1–5; per-horizon validation residual quantiles.
- **Sequence model** — LSTM (default; `--config` can switch to a small
  Transformer) over the last 10 athlete-seasons of engineered features,
  trained with the pinball loss to directly emit the 10/50/90% quantiles
  for horizons 1–5.

### Evaluation protocol

Athletes are split 70/15/15 at the *athlete* level, so no test athlete is
ever seen in training. Each test athlete is forecast from **every** career
cutoff (≥ 3 observed seasons), giving thousands of independent forecast
origins per horizon. Metrics: MAE/RMSE/MAPE on the median, mean pinball
loss, CRPS (integrated from the piecewise-linear quantile function), plus
prediction-interval coverage (PICP) and sharpness (MPIW). Interval widths are
recalibrated on validation athletes with a conformal scale factor, and
coverage significance is assessed with athlete-level Wilson intervals
(forecasts from the same athlete are correlated).

### Demo app

The demo (`scripts/serve_demo.py`) serves a web page where a runner enters
their season-by-season age, season best, race count and competition level,
and receives 1–5-season forecasts from all three models (median + 10–90%
band), plus their development archetype. It loads the artifacts produced by
the pipeline (`artifacts/model_bundle.pkl`), so run the pipeline first.

## Timeline → repository map

| Milestone | Where |
|---|---|
| Sep 23 — pull, cache, clean data | `src/data/ingest.py`, `src/data/clean.py`, `data/raw/` |
| Sep 26 — features + baseline | `src/data/features.py`, `src/models/age_curve.py`, `src/models/gbm.py` |
| Sep 28 — sequence model + comparison | `src/models/sequence.py`, `src/viz.py` (comparison figures) |
| Oct 1 — held-out evaluation + calibration | `src/evaluation/`, `reports/analysis.md` §1–2 |
| Oct 2 — clustering & archetypes | `src/clustering/archetypes.py`, `reports/analysis.md` §3 |
| Oct 4 — visuals, write-up, polished repo | `reports/figures/`, `README.md`, this repo |

## Limitations

- **Data provenance** — see *Data*: reported numbers reflect the simulated
  population unless a live API cache is present.
- **Single event** — models are trained per event; no cross-event transfer.
- **Career censoring** — early retirees are absent, biasing long-horizon
  estimates upward for continuing athletes.
- **Quantile resolution** — three quantile levels; richer distributional
  heads would sharpen CRPS.
- **No contextual covariates** — training load, injuries, coaching, altitude.

Full discussion in [`reports/analysis.md`](reports/analysis.md).

## License

MIT — see [LICENSE](LICENSE).
