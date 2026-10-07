**Table 12 (E9). The deployable model (age, sex, lead time, weekday, earlier appointments, earlier no-show rate) against the research model, holdout**

| Model | AUC-ROC | Average precision | Brier score | Sensitivity at 40% precision | AUC-ROC difference from research model |
|---|---:|---:|---:|---:|---:|
| Deployable model (six features) | 0.727 (0.719 to 0.735) | 0.330 (0.316 to 0.345) | 0.137 (0.134 to 0.139) | 0.062 (0.016 to 0.189) | -0.005 (-0.009 to -0.002) |
| Research model (all features) | 0.732 (0.724 to 0.740) | 0.339 (0.325 to 0.355) | 0.136 (0.133 to 0.138) | 0.118 (0.051 to 0.233) | n/a |

Source: `results/e9_deployable.csv`. The model in the app is not calibrated: see Table 11.
