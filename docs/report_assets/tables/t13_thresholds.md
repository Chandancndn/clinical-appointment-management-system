**Table 13 (E10). Flagging thresholds on the Kaggle holdout: how many bookings are flagged, and how many flagged are no-shows**

| Rule | Risk threshold | Share of bookings flagged | Precision (share of flagged who miss) | Sensitivity (share of no-shows flagged) |
|---|---:|---:|---:|---:|
| Risk at least 0.20 | 0.200 | 55.9% (55.2% to 56.6%) | 27.9% (27.1% to 28.7%) | 84.3% (83.1% to 85.4%) |
| Risk at least 0.25 | 0.250 | 40.3% (39.5% to 41.0%) | 30.6% (29.6% to 31.6%) | 66.6% (65.0% to 68.1%) |
| Risk at least 0.30 | 0.300 | 27.2% (26.6% to 27.9%) | 32.6% (31.5% to 33.9%) | 48.0% (46.4% to 49.7%) |
| Risk at least 0.35 | 0.350 | 16.6% (16.1% to 17.2%) | 35.0% (33.5% to 36.6%) | 31.5% (30.0% to 33.0%) |
| Risk at least 0.40 | 0.400 | 6.4% (6.1% to 6.8%) | 36.4% (33.6% to 39.0%) | 12.6% (11.5% to 13.7%) |
| Risk at least 0.45 | 0.450 | 1.6% (1.4% to 1.8%) | 43.0% (37.5% to 49.1%) | 3.7% (3.1% to 4.4%) |
| Risk at least 0.50 | 0.500 | 0.5% (0.4% to 0.6%) | 52.3% (41.7% to 63.4%) | 1.4% (1.0% to 1.9%) |
| Risk at least 0.55 | 0.550 | 0.3% (0.2% to 0.4%) | 61.5% (46.6% to 75.0%) | 1.0% (0.6% to 1.4%) |
| Risk at least 0.60 | 0.600 | 0.2% (0.1% to 0.3%) | 69.6% (52.1% to 86.4%) | 0.8% (0.5% to 1.2%) |
| Risk at least 0.65 | 0.650 | 0.2% (0.1% to 0.3%) | 70.7% (52.0% to 88.9%) | 0.7% (0.4% to 1.1%) |
| Risk at least 0.70 | 0.700 | 0.1% (0.1% to 0.2%) | 70.0% (46.2% to 92.9%) | 0.5% (0.3% to 0.8%) |
| Top share above the p50 percentile of risk | 0.226 | 48.1% (47.4% to 48.9%) | 29.5% (28.6% to 30.3%) | 76.6% (75.2% to 78.0%) |
| Top share above the p70 percentile of risk | 0.296 | 28.2% (27.6% to 28.9%) | 32.4% (31.3% to 33.6%) | 49.4% (47.8% to 51.0%) |
| Top share above the p80 percentile of risk | 0.339 | 18.9% (18.3% to 19.4%) | 34.5% (33.0% to 36.0%) | 35.2% (33.7% to 36.7%) |
| Top share above the p90 percentile of risk | 0.379 | 10.1% (9.6% to 10.5%) | 37.0% (35.0% to 39.1%) | 20.2% (18.9% to 21.4%) |
| Top share above the p95 percentile of risk | 0.405 | 5.5% (5.2% to 5.9%) | 36.8% (33.8% to 39.7%) | 11.0% (9.9% to 12.0%) |
| Chosen Medium cut (target precision 30%) | 0.190 | 58.6% (57.9% to 59.3%) | 27.4% (26.7% to 28.2%) | 86.7% (85.6% to 87.9%) |
| Chosen High cut (target precision 40%) | 0.380 | 10.0% (9.5% to 10.4%) | 37.1% (35.1% to 39.2%) | 20.0% (18.7% to 21.2%) |

Source: `results/e10_thresholds.csv`. Intervals are 95% bootstrap intervals over patients.
