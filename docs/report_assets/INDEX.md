# Report assets: index

Everything here is built from `results/` by `python -m scripts.make_report_assets`; nothing was typed by hand. Every number in the tables is listed in `docs/numbers_audit.csv`, and `python -m scripts.check_numbers` re-checks them against the results files.
Table numbers below follow this folder; renumber them to fit the report. Figures in the PNG files carry their own titles and footnotes; green and amber are light on the page background, so each figure also uses marker shapes, direct labels, and has its numbers in a table.

## Tables (CSV for editing, Markdown for pasting)

| File | Plan item | Report chapter | What it shows | Source |
|---|---|---|---|---|
| `tables/t01_dataset_summary.csv`, `tables/t01_dataset_summary.md` | E1 | 6 and 8 | The three datasets after cleaning, and how they were split | `results/dataset_summary.csv` |
| `tables/t02_cleaning_log.csv`, `tables/t02_cleaning_log.md` | E1 | 6 and 8 | Cleaning log: how many rows each rule changed | `results/cleaning_log.csv` |
| `tables/t03_baselines.csv`, `tables/t03_baselines.md` | E2 | 8 | Baselines on the locked holdouts (95% bootstrap intervals) | `results/e2_baselines.csv` |
| `tables/t04_kaggle_models.csv`, `tables/t04_kaggle_models.md` | E3 | 8 | Model comparison on the Kaggle holdout, all features (95% bootstrap intervals) | `results/e3_kaggle_models.csv` |
| `tables/t08_openml_models.csv`, `tables/t08_openml_models.md` | E5 | 8 | The same model ladder on the OpenML holdout, the last month (95% bootstrap intervals) | `results/e5_openml_models.csv` |
| `tables/t05_model_ladder.csv`, `tables/t05_model_ladder.md` | E3 | 8 | Does each step up the ladder beat the one before? Paired differences on the holdout | `results/e3_ladder.csv` |
| `tables/t06_ablation.csv`, `tables/t06_ablation.md` | E4 | 8 | Feature-group ablation: gradient boosting, patient-grouped cross-validation on the training part | `results/e4_ablation.csv` |
| `tables/t07_sms_by_lead_time.csv`, `tables/t07_sms_by_lead_time.md` | E4 | 8 | SMS reminder against no-show rate within each lead-time band (Kaggle training part) | `results/e4_sms_crosstab.csv` |
| `tables/t09_transfer.csv`, `tables/t09_transfer.md` | E6 | 8 | Transfer on the four common features (age, sex, lead time, weekday), both directions | `results/e6_transfer.csv` |
| `tables/t10_imbalance.csv`, `tables/t10_imbalance.md` | E7 | 8 | Imbalance handling on the Kaggle holdout | `results/e7_imbalance.csv` |
| `tables/t11_calibration.csv`, `tables/t11_calibration.md` | E8 | 8 | Calibration: Brier score and expected calibration error (ECE), gradient boosting with all features | `results/e8_calibration.csv` |
| `tables/t12_deployable.csv`, `tables/t12_deployable.md` | E9 | 8 | The deployable model (age, sex, lead time, weekday, earlier appointments, earlier no-show rate) against the research model, holdout | `results/e9_deployable.csv` |
| `tables/t13_thresholds.csv`, `tables/t13_thresholds.md` | E10 | 8 | Flagging thresholds on the Kaggle holdout: how many bookings are flagged, and how many flagged are no-shows | `results/e10_thresholds.csv` |
| `tables/t14_risk_bands.csv`, `tables/t14_risk_bands.md` | E10 | 8 | The Low, Medium and High bands shown to staff | `results/e10_thresholds.csv` |
| `tables/t15_policy_tradeoffs.csv`, `tables/t15_policy_tradeoffs.md` | S2 | 8 | Booking policies: the trade-off table (no weighted score is computed) | `results/s2_policy_tradeoffs.csv` |
| `tables/t16_probability_error.csv`, `tables/t16_probability_error.md` | S3 | 8 | Sensitivity to error in the predicted risk (a constant added to every predicted probability) | `results/s3_probability_error.csv` |
| `tables/t17_base_rate.csv`, `tables/t17_base_rate.md` | S4 | 8 | Sensitivity to the clinic's base no-show rate | `results/s4_base_rate.csv` |
| `tables/t18_service_variability.csv`, `tables/t18_service_variability.md` | S5 | 8 | Sensitivity to the spread of consultation times (same mean) | `results/s5_service_variability.csv` |
| `tables/t19_value_of_prediction.csv`, `tables/t19_value_of_prediction.md` | S6 | 8 | Value of prediction at equal overbooking: the same number of double-booked slots, chosen four ways | `results/s6_value_of_prediction.csv` |
| `tables/t20_service_time_fit.csv`, `tables/t20_service_time_fit.md` | S1 | 7 and 8 | Consultation-time distributions fitted to the Hangu data (minutes) | `results/sim_service_time_fit.csv` |
| `tables/t21_test_summary.csv`, `tables/t21_test_summary.md` | Chapter 7 | 7 | Test summary: tests run and passed, in total, per database engine and per test file | `results/test_summary.csv`, `results/test_summary.txt` |

## Figures (PNG)

| File | Plan item | Report chapter | What it shows | Source | How it was made |
|---|---|---|---|---|---|
| `figures/f01_e1_overall_noshow_rate.png` | E1 | 6 and 8 | Overall no-show rate in the two appointment datasets. | `results/e1_eda_rates.csv`, `results/figures/e1_overall_noshow_rate.png` | copied unchanged from results/figures |
| `figures/f02_e1_noshow_by_lead_time.png` | E1 | 6 and 8 | No-show rate by days between booking and appointment, both datasets. | `results/e1_eda_rates.csv`, `results/figures/e1_noshow_by_lead_time.png` | copied unchanged from results/figures |
| `figures/f03_e1_noshow_by_weekday.png` | E1 | 6 and 8 | No-show rate by appointment weekday, both datasets. | `results/e1_eda_rates.csv`, `results/figures/e1_noshow_by_weekday.png` | copied unchanged from results/figures |
| `figures/f04_e1_noshow_by_age_band.png` | E1 | 6 and 8 | No-show rate by age band, both datasets. | `results/e1_eda_rates.csv`, `results/figures/e1_noshow_by_age_band.png` | copied unchanged from results/figures |
| `figures/f05_e3_roc.png` | E3 | 8 | ROC curves of the three models on the Kaggle holdout. | `results/e3_kaggle_models.csv`, `results/figures/e3_roc.png` | result image reused (curve points were not saved); footnote redrawn |
| `figures/f06_e3_pr.png` | E3 | 8 | Precision-recall curves of the three models on the Kaggle holdout. | `results/e3_kaggle_models.csv`, `results/figures/e3_pr.png` | result image reused (curve points were not saved); footnote redrawn |
| `figures/f07_e4_ablation.png` | E4 | 8 | Change in cross-validated AUC-ROC when a feature group is removed, and when only that group is kept. | `results/e4_ablation.csv` | redrawn from the CSV |
| `figures/f08_e6_transfer.png` | E6 | 8 | AUC-ROC within the target dataset and after transfer, both directions, common features only. | `results/e6_transfer.csv` | redrawn from the CSV |
| `figures/f09_e8_reliability.png` | E8 | 8 | Reliability diagram on the Kaggle holdout, ten equal-count bins. | `results/e8_reliability.csv`, `results/e8_calibration.csv` | redrawn from the CSV |
| `figures/f10_e10_thresholds.png` | E10 | 8 | Precision, sensitivity and share of bookings flagged at each risk threshold, with the Medium and High cuts. | `results/e10_thresholds.csv` | redrawn from the CSV |
| `figures/f11_s2_tradeoffs.png` | S2 | 8 | Patients served against mean wait and against doctor overtime for every booking policy. | `results/s2_policy_tradeoffs.csv` | redrawn from the CSV |
| `figures/f12_s3_probability_error.png` | S3 | 8 | Double-booked slots, wait and overtime when the predicted risk is shifted by a constant. | `results/s3_probability_error.csv` | redrawn from the CSV |
| `figures/f13_s4_base_rate.png` | S4 | 8 | Patients served, wait and overtime when the clinic's base no-show rate is scaled. | `results/s4_base_rate.csv` | redrawn from the CSV |
| `figures/f14_s5_service_variability.png` | S5 | 8 | Wait and overtime when the spread of consultation times is scaled at the same mean. | `results/s5_service_variability.csv` | redrawn from the CSV |
| `figures/f15_s6_value_of_prediction.png` | S6 | 8 | Collisions, waiting and overtime against the number of double-booked slots, by how the slots are chosen. | `results/s6_value_of_prediction.csv`, `results/figures/s6_value_of_prediction.png` | copied unchanged from results/figures |
| `figures/f16_s6_paired_differences.png` | S6 | 8 | Paired differences in waiting and overtime between risk-based, evenly spaced, random and oracle choices at equal overbooking. | `results/s6_value_of_prediction.csv` | redrawn from the CSV |
| `figures/f17_service_time_fit.png` | S1 | 7 and 8 | Hangu consultation times with the lognormal and gamma fits. | `results/sim_service_time_fit.csv`, `results/figures/sim_service_time_fit.png` | copied unchanged from results/figures |
| `figures/f18_service_time_qq.png` | S1 | 7 and 8 | QQ plots of the Hangu consultation times against the fitted lognormal and gamma. | `results/sim_service_time_fit.csv`, `results/figures/sim_service_time_qq.png` | result image reused (the quantiles were not saved); footnote redrawn |

## Not produced here

The architecture diagram, ER diagram, booking-flow diagram and the screenshots of each role's pages are drawn or taken by the team (PLAN section 8); `docs/DEMO_SCRIPT.md` lists the pages to capture.
