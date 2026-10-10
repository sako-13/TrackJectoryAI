# Career-Forecast Analysis — 100m

**Data source:** synthetic  
**Athletes evaluated:** 2,228  
**Athlete-seasons:** 22,004  
**Evaluation:** every test athlete was forecast from every career cutoff (≥3 observed seasons); horizons 1–5 seasons; metrics aggregated per horizon. The split is at the *athlete* level, so no test athlete was seen during training.

## 1. Overall accuracy

| Model | MAE (s) | RMSE (s) | CRPS (s) | MAPE (%) |
|---|---:|---:|---:|---:|
| age_curve | 0.150 | 0.204 | 0.127 | 1.27 |
| gbm | 0.103 | 0.147 | 0.089 | 0.87 |
| lstm | 0.090 | 0.133 | 0.078 | 0.77 |

**Best model: lstm.**

### Accuracy by horizon (best model)

| Horizon | MAE (s) | RMSE (s) | CRPS (s) |
|---:|---:|---:|---:|
| 1 | 0.049 | 0.075 | 0.043 |
| 2 | 0.073 | 0.107 | 0.064 |
| 3 | 0.096 | 0.134 | 0.085 |
| 4 | 0.126 | 0.169 | 0.107 |
| 5 | 0.152 | 0.200 | 0.132 |

## 2. Prediction-interval calibration

Each model emits 10%/50%/90% quantiles, i.e. an 80% prediction interval. Interval widths are recalibrated with a per-horizon conformal scale factor fitted on the *validation* athletes, so the reported coverage on test athletes is honest. A calibrated model covers the true season best 80% of the time (within the Wilson 95% confidence band).

| Model | Empirical coverage | Nominal | 95% CI | Mean width (s) | Calibrated? |
|---|---:|---:|---:|---:|---|
| age_curve | 0.840 | 0.80 | [0.769, 0.852] | 0.528 | yes |
| gbm | 0.837 | 0.80 | [0.761, 0.845] | 0.368 | yes |
| lstm | 0.818 | 0.80 | [0.727, 0.816] | 0.322 | yes |

### Coverage by horizon

| Model | H=1 | H=2 | H=3 | H=4 | H=5 |
|---|---:|---:|---:|---:|---:|
| age_curve | 0.84 | 0.83 | 0.84 | 0.85 | 0.85 |
| gbm | 0.83 | 0.83 | 0.83 | 0.84 | 0.84 |
| lstm | 0.82 | 0.82 | 0.81 | 0.83 | 0.82 |

## 3. Development archetypes

Career trajectories (peak age, improvement rate, career length, gains) were standardized and clustered with K-Means (k=4). Archetypes are labelled from the cluster centers: low peak age → *Early Peaker*, high peak age → *Late Bloomer*, high improvement rate → *Rapid Riser*, else *Steady Developer*.

| Archetype | Athletes | Peak age | Peak best (s) | Total gain (s) | Improvement rate (s/yr) |
|---|---:|---:|---:|---:|---:|
| Early Peaker | 590 | 23.4 | 12.41 | 0.47 | 0.09 |
| Late Bloomer | 510 | 25.1 | 11.81 | 0.94 | 0.20 |
| Steady Developer | 595 | 24.1 | 10.96 | 0.49 | 0.09 |
| Rapid Riser | 533 | 23.7 | 11.76 | 0.99 | 0.15 |

## 4. Limitations

- **Data provenance:** World Athletics does not publish an official public API. The ingestion client targets the site's unofficial endpoints; when those are unreachable the pipeline trains on a realistic *synthetic* population with the same schema, so all reported numbers reflect the simulated population unless a live cache from the API is present in `data/raw/`.
- **Single-event scope:** models are trained per event; cross-event transfer is not attempted.
- **Career censoring:** athletes who retire early are absent from the data, which biases long-horizon estimates upward for continuing athletes.
- **Quantile resolution:** intervals are built from only three quantile levels; richer distributional heads would sharpen CRPS.
- **No contextual covariates:** training load, injuries, coaching changes and altitude are not available.
