**Table 11 (E8). Calibration: Brier score and expected calibration error (ECE), gradient boosting with all features**

| Data | Calibration method | Brier score | Expected calibration error | AUC-ROC |
|---|---|---:|---:|---:|
| Training part, grouped CV | None (uncalibrated) | 0.14476 | 0.0021 | 0.7373 |
| Training part, grouped CV | Platt scaling (sigmoid) | 0.14476 | 0.0018 | 0.7375 |
| Training part, grouped CV | Isotonic regression | 0.14482 | 0.0024 | 0.7373 |
| Locked holdout, scored once | None (uncalibrated) | 0.1357 (0.1328 to 0.1385) | 0.0124 | 0.7321 (0.7237 to 0.7401) |

Source: `results/e8_calibration.csv`. The calibrator is chosen by grouped-CV Brier score on the training part; a tie between methods means calibration adds nothing here.
