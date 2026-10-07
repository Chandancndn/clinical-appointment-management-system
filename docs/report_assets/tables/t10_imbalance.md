**Table 10 (E7). Imbalance handling on the Kaggle holdout**

| Base model | Strategy | AUC-ROC | Average precision | Brier score | Mean predicted risk |
|---|---|---:|---:|---:|---:|
| Logistic regression | None | 0.718 (0.710 to 0.726) | 0.316 (0.304 to 0.329) | 0.138 (0.135 to 0.141) | 0.199 |
| Logistic regression | Class weights | 0.718 (0.710 to 0.726) | 0.314 (0.302 to 0.327) | 0.215 (0.212 to 0.217) | 0.442 |
| Logistic regression | Threshold tuning only | 0.718 (0.710 to 0.726) | 0.316 (0.304 to 0.329) | 0.138 (0.135 to 0.141) | 0.199 |
| Logistic regression | Random under-sampling | 0.717 (0.709 to 0.726) | 0.313 (0.301 to 0.326) | 0.215 (0.212 to 0.217) | 0.443 |
| Logistic regression | SMOTE | 0.717 (0.709 to 0.725) | 0.313 (0.300 to 0.325) | 0.214 (0.211 to 0.216) | 0.439 |
| Logistic regression | NearMiss | 0.566 (0.556 to 0.577) | 0.203 (0.195 to 0.213) | 0.460 (0.453 to 0.466) | 0.678 |
| Gradient boosting | None | 0.732 (0.724 to 0.740) | 0.339 (0.325 to 0.355) | 0.136 (0.133 to 0.138) | 0.197 |
| Gradient boosting | Class weights | 0.733 (0.724 to 0.741) | 0.341 (0.327 to 0.356) | 0.207 (0.204 to 0.209) | 0.432 |
| Gradient boosting | Threshold tuning only | 0.732 (0.724 to 0.740) | 0.339 (0.325 to 0.355) | 0.136 (0.133 to 0.138) | 0.197 |
| Gradient boosting | Random under-sampling | 0.731 (0.723 to 0.739) | 0.338 (0.325 to 0.353) | 0.210 (0.208 to 0.213) | 0.438 |
| Gradient boosting | SMOTE | 0.720 (0.712 to 0.727) | 0.320 (0.306 to 0.333) | 0.156 (0.154 to 0.158) | 0.307 |
| Gradient boosting | NearMiss | 0.635 (0.626 to 0.645) | 0.247 (0.237 to 0.259) | 0.429 (0.423 to 0.435) | 0.660 |

Source: `results/e7_imbalance.csv`. The true no-show rate on the holdout is 0.185; a mean predicted risk far from it means the probabilities are distorted.
