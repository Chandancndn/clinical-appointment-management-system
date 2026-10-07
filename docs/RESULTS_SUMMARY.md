# CAMS results summary

Every number in this file was read from a file in `results/` when it was built (`python -m scripts.make_report_assets`). The same numbers are listed in `docs/numbers_audit.csv`, and `python -m scripts.check_numbers` fails if any of them no longer matches its results file. Each section ends with the report chapter it belongs in. The results are frozen: no experiment was re-run for the report. The tables and figures mentioned are in `docs/report_assets/` (see `INDEX.md` there for the numbering).

In every table, a 95% interval is a bootstrap over patients on the locked holdout unless the text says otherwise. The positive class is always a no-show.

## Experiments on the prediction model

### E1 Dataset profile and cleaning

The Kaggle file holds 110,527 appointments. Cleaning removed 16 of them (the cleaning log counts each rule: 5 fractional patient IDs, 6 impossible ages, 5 appointments dated before they were booked), leaving 110,511 appointments from 62,291 patients between 2016-04-29 and 2016-06-08. 20.2% were no-shows. 34.9% of appointments are booked for the same day and only 4.8% of those are missed, against 33.9% for bookings made 15 to 30 days ahead. The OpenML file has 61,214 appointments, 60,990 after removing exact duplicates, with 21.0% no-shows. The Hangu file has 6,637 consultations from 381 half-day sessions, with a median of 12.1 minutes and a mean of 13.4 minutes. The Kaggle holdout is the latest 5 of 27 appointment dates (21,984 appointments); OpenML holds out its last month. No patient appears on both sides of any cross-validation fold (0 patients on both sides).

Report chapter: Chapter 6 (data design) and Chapter 8 (Results: Tables 1 and 2, Figures 1 to 4).

### E2 Baselines

Always predicting "show" looks good on accuracy (0.815 on the Kaggle holdout) but is a coin flip on AUC-ROC (0.500), and its average precision is just the no-show rate (0.185). That is why accuracy is never the headline. A rule that uses only a patient's earlier no-show rate reaches an AUC-ROC of 0.518 (0.511 to 0.525): little signal there, partly because patients average only 1.77 appointments in this file.

Report chapter: Chapter 8 (Results, Table 3).

### E3 Model comparison on the Kaggle data

With every feature the dataset offers, the holdout AUC-ROC is 0.718 (0.710 to 0.726) for logistic regression, 0.730 (0.722 to 0.738) for the random forest and 0.732 (0.724 to 0.740) for gradient boosting. All three are well below the 0.85 level that would signal leakage, and a model trained on shuffled labels scores 0.506, as it should. Moving from the earlier-rate rule to logistic regression and then to the random forest are clear improvements, with changes in AUC-ROC of +0.200 (+0.190 to +0.210) and +0.012 (+0.007 to +0.018). Gradient boosting over the random forest is not clearly better: the change is +0.002 (-0.001 to +0.004), so the two are statistically close. Brier scores are 0.136 for the random forest and 0.136 for gradient boosting; the logistic regression's 0.215 is not comparable because it uses class weights. At 40% precision gradient boosting finds 0.118 of the no-shows, and at 50% precision 0.022: the model ranks patients usefully but cannot be confident about many of them.

Report chapter: Chapter 8 (Results, Table 4, Table 5, Figures 5 and 6).

### E4 Which features matter

With patient-grouped cross-validation on the training part, the reference model with all features has an AUC-ROC of 0.738. Removing lead time and weekday changes the AUC-ROC the most, by -0.114, and those two alone keep an AUC-ROC of 0.706. Removing age and sex changes it by -0.011, removing patient history by -0.005 and removing the SMS reminder by -0.002; history alone reaches only 0.545. The SMS reminder needs care: it was sent selectively, to 4.5% of bookings made one to three days ahead but 55.6% of those made four to seven days ahead. Within every lead-time band that has both groups, patients who received an SMS missed less: the no-show rate was -2.0 percentage points different (SMS minus none) in the shortest band and -8.1 in the longest. That is an association, not proof that the reminder causes it.

Report chapter: Chapter 8 (Results, Tables 6 and 7, Figure 7).

### E5 The same ladder on OpenML

On the OpenML holdout (its last month) the AUC-ROC is 0.612 (0.602 to 0.623) for logistic regression, 0.648 (0.638 to 0.658) for the random forest and 0.646 (0.636 to 0.655) for gradient boosting, so the random forest and gradient boosting are statistically close here too, and both are clearly above logistic regression. These scores are lower than on Kaggle. The OpenML file does not say where it comes from and has no patient ID, so it cannot say why; it is a second dataset that shows the same ordering of models, not a replication of the Kaggle numbers.

Report chapter: Chapter 8 (Results, Table 8).

### E6 Does it transfer?

Using only the four features both datasets share (age, sex, lead time, weekday), a model trained on Kaggle scores 0.529 (logistic regression) and 0.538 (gradient boosting) on OpenML, against 0.540 and 0.568 for models trained on OpenML itself. The AUC-ROC drops (positive means the transferred model is worse) are +0.010 (-0.001 to +0.023) and +0.030 (+0.018 to +0.042); the first interval includes zero. In the other direction a model trained on OpenML scores 0.640 and 0.608 on the Kaggle holdout, against 0.713 and 0.723: drops of +0.072 (+0.062 to +0.082) and +0.114 (+0.104 to +0.124). The findings transfer only partly: the main Kaggle signal, lead time, behaves differently in the two datasets (same-day bookings are missed 4.8% of the time on Kaggle but 17.3% on OpenML), which is a plausible reason but one these data cannot confirm.

Report chapter: Chapter 8 (Results, Table 9, Figure 8).

### E7 Imbalance handling

Class weights and threshold tuning leave the ranking (AUC-ROC) unchanged: for logistic regression it is 0.718 without class weights and 0.718 with them. Class weights do distort the probabilities, though: the mean predicted risk rises to 0.442 against a true rate of 0.185, and the Brier score from 0.138 to 0.215. Resampling is worse. SMOTE lowers gradient boosting's AUC-ROC to 0.720 and raises its Brier score to 0.156; NearMiss collapses the AUC-ROC to 0.566 (logistic regression) and 0.635 (gradient boosting). The final model therefore uses no resampling; it stays in the report only as an ablation.

Report chapter: Chapter 8 (Results, Table 10).

### E8 Calibration

On patient-grouped cross-validation the Brier score of gradient boosting is 0.1448 with no calibration, 0.1448 with Platt scaling and 0.1448 with isotonic regression: a tie to four decimals. Because neither calibrator lowers the Brier score, the selection rule keeps the model uncalibrated, and the model used later (in the deployable model, the simulation and the app) is uncalibrated. On the holdout its expected calibration error is 0.0124 and its Brier score 0.1357 (0.1328 to 0.1385). The reliability diagram sits close to the diagonal, with the highest-risk bin slightly over-predicting (predicted 40.9%, observed 37.4%).

Report chapter: Chapter 8 (Results, Table 11, Figure 9).

### E9 The deployable model

The model the app and the simulation use sees only what a clinic can collect: age, sex, lead time, weekday, earlier appointments and earlier no-show rate. On the holdout it reaches an AUC-ROC of 0.727 (0.719 to 0.735), average precision 0.330 (0.316 to 0.345) and Brier score 0.137 (0.134 to 0.139), against 0.732 (0.724 to 0.740) for the research model with every feature. Giving up the features the app cannot collect changes the AUC-ROC by -0.005 (-0.009 to -0.002): small, but the interval excludes zero. The loss shows more at high precision: sensitivity at 40% precision is 0.062 (0.016 to 0.189) against 0.118 (0.051 to 0.233). The model is uncalibrated (see E8), was trained on Kaggle data only, and uses age and sex.

Report chapter: Chapter 8 (Results, Table 12) and Chapter 6 (the model that the app uses).

### E10 Risk thresholds and bands

The model card sets the Medium cut at 0.19 and the High cut at 0.38. On the holdout the bands are: Low, 41.4% of bookings, of which 5.9% are no-shows; Medium, 48.6% of bookings and 25.4% no-shows; High, 10.0% of bookings and 37.1% no-shows. The cuts were chosen on out-of-fold predictions for the training part, aiming at a precision near 30% for Medium and 40% for High; on the holdout they reach 27.4% and 37.1%, a little short of those aims. The High band holds 20.0% of all no-shows, and Medium and High together hold 86.7%. So a High flag roughly doubles the chance of a no-show compared with the clinic average (18.5% on the holdout), and a Low flag cuts it to about a third, but even High bookings attend more often than not. That is why the flag is advisory.

Report chapter: Chapter 8 (Results, Tables 13 and 14, Figure 10) and Chapter 6 (the risk flag).

## Experiments on the booking simulation

### S1 Sanity tests and the consultation-time fit

The simulation was checked before it was used: 31 tests in `tests/test_sim.py` and 16 in `tests/test_sim_experiments.py` pass. They cover the cases the plan lists (everyone attends and service equals the slot: no wait and no overtime; nobody attends: nobody served and idle time equals the session; a hand-worked three-patient case; the same seed gives the same result). Consultation times are modelled as lognormal rather than gamma because of a lower AIC (41,027.6 against 41,268.5); the QQ plot shows the lognormal following the data closely, with a slightly heavier tail in the data than the fit at the top end. Hangu's median consultation of 12.1 minutes gives a default slot of 15 minutes (the median rounded up to the next five) and a session of 240 minutes, which is 16 slots. The file has no consultation shorter than 3 minutes, so its short end is missing.

Report chapter: Chapter 7 (Implementation and Testing) and Chapter 8 (Results, Table 20, Figures 17 and 18).

### S2 Policy trade-offs

Every setting uses 1,000 replications, with seed 42 plus the replication number. With no overbooking (P0) a session serves 12.75 patients on average, with a mean wait of 2.9 minutes, doctor overtime of 3.7 minutes and doctor idle time of 72.1 minutes. Doubling every second slot (P1, k = 2) serves 19.16 patients but the wait becomes 21.4 minutes, overtime 37.3 minutes and idle time falls to 19.7 minutes. A risk threshold at p90 (P2) double-books 1.52 slots per session and serves 13.99 patients with a wait of 5.2 minutes and overtime of 6.6 minutes. Every extra patient served costs waiting and overtime, and no policy is best on all of them, so the report gives the trade-off table and does not combine it into one score: any weighting would be a choice made by the reader.

Report chapter: Chapter 8 (Results, Table 15, Figure 11).

### S3 If the predicted risk is wrong

A constant error in every predicted risk changes how many slots a fixed threshold double-books. For the P2 threshold of 0.3, unshifted, 4.36 slots are double-booked and the mean wait is 11.0 minutes. Overestimating the risk by 0.10 raises that to 8.94 slots, a wait of 24.8 minutes and overtime of 42.5 minutes. Underestimating it by 0.10 cuts it to 0.93 slots, a wait of 4.3 minutes and 13.51 patients served, barely more than the 12.75 of P0. The effect is lopsided: overestimating adds more waiting than underestimating removes, which is in line with the arXiv preprint by Amalina and An (2026; not peer reviewed), though that comparison is of direction only.

Report chapter: Chapter 8 (Results, Table 16, Figure 12).

### S4 If the clinic's no-show rate is different

Scaling every patient's no-show probability by one half gives a base rate of 10.2%, and by one and a half 30.6%. Without overbooking a session then serves 14.38 patients at the low rate and 11.18 at the high rate. Doubling every second slot costs a wait of 30.9 minutes at the low rate (patients come more often) and 14.4 minutes at the high rate. The fixed risk threshold of 0.3 adapts: it double-books 0.03 slots per session at the low rate and 8.94 at the high rate, whereas the uniform and percentile policies double-book the same number whatever the rate. A policy that does not look at risk has to be re-tuned for each clinic.

Report chapter: Chapter 8 (Results, Table 17, Figure 13).

### S5 If consultation times vary more

With the mean consultation time held at 13.4 minutes, the standard deviation is scaled from 3.1 to 6.1 to 9.2 minutes. The same patients attend in every setting, so patients served do not change, but waiting does: with no overbooking the mean wait is 0.81, 2.90 and 5.56 minutes, and with every third slot doubled (P1, k = 3) it is 10.1, 13.1 and 16.8 minutes. The more variable the consultations, the more overbooking costs.

Report chapter: Chapter 8 (Results, Table 18, Figure 14).

### S6 The value of prediction at equal overbooking (the headline)

More double-booking always serves more patients, so the fair question is whether choosing the slots by predicted risk beats choosing them evenly or at random when the number of double-booked slots is the same. In this experiment each comparison double-books exactly as many slots as the policy it is matched to, so patients served are identical across the ways of choosing (about 0.80 extra patients per double-booked slot) and only waiting, overtime and collisions can differ. The P2 settings are named by the percentile of predicted risk used as the cut: p50 is a low cut that double-books many slots, p95 a high cut that double-books few. Differences below are paired over replications, in minutes, and below zero favours the risk-based choice.

Against evenly spaced slots, the risk threshold lowers the mean wait by a small amount in most settings: -0.412 (-0.678 to -0.145) at the 0.3 threshold, -0.528 (-0.808 to -0.249) at p70, -0.456 (-0.674 to -0.237) at p80, -0.572 (-0.734 to -0.409) at p90 and -0.370 (-0.489 to -0.251) at p95. At p50 the interval includes zero (-0.148 (-0.456 to +0.161)). The overtime differences against evenly spaced slots are mostly inconclusive, and at p95 the risk threshold is slightly worse (+0.240 (+0.025 to +0.455)). Against random slots the gain is larger, for example -1.301 (-1.688 to -0.914) at p50.

The share of double-booked slots in which both patients attend, the collision that overbooking tries to avoid, is 50.7% with the risk threshold at 0.3, against 63.2% with evenly spaced slots, 63.3% with random slots and 26.8% for an oracle that knows who will not come. An oracle would do far better on waiting too (-2.477 (-2.695 to -2.259) at p80), so the model captures only a small part of what perfect knowledge of attendance would give.

Matched to the uniform policies instead (the highest-risk slots against every k-th slot), the picture is mixed. The wait falls at k = 3, by -0.578 (-0.863 to -0.292), and at k = 6, by -0.807 (-0.999 to -0.616), but not clearly at k = 2 (+0.136 (-0.195 to +0.467)) or k = 4 (+0.148 (-0.111 to +0.407)), and the overtime at k = 6 is higher (+0.563 (+0.238 to +0.888)).

In short, prediction adds a modest, not uniform, benefit over spacing the extra patients evenly, and a clearer one over picking at random. This is also the best case for the model, because in the simulation the true no-show risk is the model's own risk.

Report chapter: Chapter 8 (Results, Table 19, Figures 15 and 16), with the discussion in Chapter 10.

## Limitations to state plainly

- **No real clinic data.** The model learned from a Brazilian public-clinic dataset (Vitória, Kaggle), consultation times come from a private traditional-medicine clinic in China (Hangu), and the app runs on synthetic records. The simulation shows a method and its trade-offs, not clinical outcomes.
- **Assumed clinic behaviour.** One doctor, one half-day session, slots of equal length, patients who arrive on time and are seen in slot order, no doctor time used by a no-show, and an extra patient whose attendance is drawn from the same risk model. In the simulation the model's risk is the true risk, so S6 shows the most prediction can help.
- **Short data.** The Kaggle data covers about six weeks (2016-04-29 to 2016-06-08) and OpenML months 1 to 4 of one year, so seasonal patterns are not captured, and the Kaggle holdout is only 5 dates.
- **Advisory use only.** The flag is shown to doctors and admins as Low, Medium or High, never to patients; the booking logic never reads it, so booking is unaffected if the model is missing or fails; and no patient is refused or moved because of it. The model has not been tested on live clinic traffic.
- **Age and sex are in the model.** This raises fairness questions. Removing them changes the AUC-ROC by -0.011 in E4, so the model would remain usable without them; because the flag only informs staff, nobody is denied a slot on that basis.
- **Modest accuracy.** The deployable model's AUC-ROC is 0.727, and even a High flag is wrong more often than right (E10). It is a prompt for staff attention, not a decision.
- **No calibrator.** E8 found no calibration method that lowered the Brier score, so the model is uncalibrated; its calibration error on the holdout is small but not zero.
- **Weak transfer.** E6 shows that findings on one dataset transfer only partly to the other, so a clinic would need its own data and a re-run of the pipeline.
- **Associations only.** The SMS reminder result is an association; reminders were sent selectively.
- **A preprint.** Amalina and An (2026) is an arXiv preprint and has not been peer reviewed.

## How the numbers were produced

```
python -m scripts.run_test_summary       # tests on SQLite and MySQL -> results/test_summary.csv and .txt
python -m scripts.make_report_assets     # tables, figures, this file, docs/numbers_audit.csv
python -m scripts.check_numbers --fresh  # every quoted number against its results file; docs/ against a fresh build
```

Each result file has an entry in `results/manifest.json` (script, seed, data file hashes, date, git commit).
