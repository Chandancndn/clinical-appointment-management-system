**Table 8 (E5). The same model ladder on the OpenML holdout, the last month (95% bootstrap intervals)**

| Model | AUC-ROC | Average precision | Brier score | Sensitivity at 30% precision | Sensitivity at 40% precision | Sensitivity at 50% precision |
|---|---:|---:|---:|---:|---:|---:|
| Always predict show | 0.500 (0.500 to 0.500) | 0.220 (0.213 to 0.226) | 0.220 (0.213 to 0.226) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| Logistic regression | 0.612 (0.602 to 0.623) | 0.350 (0.334 to 0.366) | 0.235 (0.234 to 0.237) | 0.394 (0.310 to 0.477) | 0.182 (0.156 to 0.218) | 0.125 (0.110 to 0.150) |
| Random forest | 0.648 (0.638 to 0.658) | 0.392 (0.377 to 0.408) | 0.158 (0.155 to 0.162) | 0.580 (0.501 to 0.668) | 0.246 (0.214 to 0.308) | 0.169 (0.144 to 0.190) |
| Gradient boosting | 0.646 (0.636 to 0.655) | 0.391 (0.376 to 0.407) | 0.158 (0.155 to 0.162) | 0.542 (0.466 to 0.631) | 0.259 (0.220 to 0.299) | 0.164 (0.141 to 0.184) |

Source: `results/e5_openml_models.csv`.
