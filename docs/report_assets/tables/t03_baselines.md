**Table 3 (E2). Baselines on the locked holdouts (95% bootstrap intervals)**

| Dataset | Model | AUC-ROC | Average precision | Brier score | Accuracy |
|---|---|---:|---:|---:|---:|
| Kaggle (Vitória, Brazil) | Always predict show | 0.500 (0.500 to 0.500) | 0.185 (0.180 to 0.190) | 0.185 (0.180 to 0.190) | 0.815 |
| Kaggle (Vitória, Brazil) | Earlier no-show rate rule | 0.518 (0.511 to 0.525) | 0.196 (0.189 to 0.203) | 0.211 (0.206 to 0.216) | 0.733 |
| OpenML Medical-Appointment | Always predict show | 0.500 (0.500 to 0.500) | 0.220 (0.213 to 0.226) | 0.220 (0.213 to 0.226) | 0.780 |

Source: `results/e2_baselines.csv`. Accuracy is shown only to make a point: always predicting show scores high on it and is useless.
