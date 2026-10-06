# CAMS: Clinic Appointment Management System

V-semester CSE mini project, SIT Tumakuru. Guide: Dr. Joydeep Roy.
Team: Hrishikesh G Mahuli, Aadyantha G Nadig, Pranav P Patel, Mohammad Atif.
Stack: Python, Flask, SQLAlchemy, HTML/CSS/JS, MySQL; scikit-learn for models, SimPy for simulation. Roles: Patient, Doctor, Admin.

The guide allows and encourages ML and simulation. No real clinic data exists, so all learning uses public datasets and all clinic behaviour in the simulation is assumed and stated.

**Full design and build order: `docs/PLAN.pdf`** (the "CAMS Master Plan: Design to Final Project" document, 24 pages; read it with page ranges, e.g. pages 1-8). At the start of every session, read this file and the PLAN section for the milestone being worked on. This file holds the rules; PLAN holds the design.

## Hard rules

1. **No double-booking.** The same doctor, date and time can never be confirmed twice. Enforce it in the database, not only in Python:
   - `slots` has UNIQUE(doctor_id, slot_date, slot_time).
   - `bookings.confirmed_slot_id` is a generated column equal to `slot_id` for every status except `cancelled` (NULL when cancelled), with its own UNIQUE constraint. A completed or no-show appointment keeps its slot; only cancelling frees it.
   - The app just tries to INSERT and treats a duplicate-key error as "slot taken" (HTTP 409). Never check-then-insert.
   - Reschedule = insert the new booking and cancel the old one in one transaction; if the new slot is taken, everything rolls back.
   - Fallback if the MySQL version cannot do generated columns: a `slot_holds` table with `slot_id` as primary key, inserted and deleted in the same transaction as the booking. Keep the same tests.
   - Tests: two concurrent requests for one slot give exactly one success; six concurrent requests across 25 slots give one winner per slot; a direct SQL insert of a second active booking fails. Run them on SQLite **and on real MySQL**. Write these tests first and show them failing.
2. **The app never overbooks.** Overbooking is studied in the simulation only. In the app, a high-risk booking is shown to staff as an advisory flag (and may feed a standby list), but the slot rule above is never relaxed. The risk model is never read by booking logic; if it is missing or fails, booking is unaffected.
3. **Every number in the report comes from a script** in this repo that writes to `results/`. Never type, estimate or invent a metric. If a result is missing, run the experiment. Each result file gets a `results/manifest.json` entry: seed, data file hash, date, git commit. Maintain `docs/numbers_audit.csv` (number, results file, column) and a script that fails if any quoted value no longer matches.
4. **No real patient data anywhere.** The app is seeded with synthetic records (`db/seed_synthetic.py`, which never reads the public datasets). Public datasets are used for training only.
5. **Never commit raw datasets.** Keep them in `data/raw/` (gitignored) and document the download steps in `data/README.md`.
6. **Fix random seeds** and record them with each result.
7. **Tests first** for booking, leakage and simulation sanity.
8. **If a hard rule seems to be in the way, stop and ask.** Do not bend it.

## Data model

| Table | Key columns |
|---|---|
| `users` | id, name, email (unique), password_hash, role, created_at |
| `patient_profiles` | user_id (unique), date_of_birth, sex |
| `doctors` | id, user_id (unique), specialization, slot_minutes |
| `slots` | id, doctor_id, slot_date, slot_time (UNIQUE on doctor, date, time) |
| `bookings` | id, slot_id, patient_id, status (confirmed, completed, no_show, cancelled), reason, created_at, cancelled_at, cancelled_by, confirmed_slot_id (generated) |
| `risk_scores` (milestone M7) | booking_id, no_show_probability, model_version, scored_at |

Passwords hashed, CSRF token on every POST, role check on every route, patients get a 404 for other patients' bookings, all SQL through SQLAlchemy bound parameters. Doctors and admins mark past appointments completed or no_show.

## Datasets (decided)

| Role | Dataset | Notes |
|---|---|---|
| Primary (develop and report) | Kaggle "Medical Appointment No Shows" (`joniarroba/noshowappointments`), 110,527 appointments, 14 variables, Vitória (Brazil), about 20% no-shows, appointments 29 Apr to 8 Jun 2016 | No doctor ID, no service times. Clean: negative age, AppointmentDay before ScheduledDay, duplicates, column typos, float PatientId |
| Second (check that findings hold) | OpenML "Medical-Appointment" (id 43617), about 61K rows, Jan to Apr 2017, licence GPL 2 | Has specialty, booking channel (call centre/personal/web), hour, lead days. Origin and country not stated. No patient ID listed. Use the discrete month/weekday/hour columns, not the cosine ones |
| Consultation times (simulation only) | Hangu open data (`github.com/fenghaolin/HanguData`, Zenodo doi 10.5281/zenodo.7484205), 6,637 consultations from 381 half-day sessions of a single physician (2018 to 2019), licence CC BY-NC-SA 4.0 | Use the file with `ServTime` in seconds; record which file in `data/README.md`. Cite Feng et al. (2023), Data 8(3), 47, doi:10.3390/data8030047 |

**Target coding trap.** In OpenML, `show` is 1 for attended. In Kaggle, `No-show` is "Yes" for missed. In all code the positive class is **no-show = 1**. Convert OpenML with `no_show = 1 - show`.

Kaggle needs a login, so the user downloads raw files by hand. Inspect real columns first and report any mismatch with these descriptions before writing code. Cleaning functions log how many rows each rule changed to `results/cleaning_log.csv`.

## Prediction plan

Two kinds of model:
- **Research models:** every feature a dataset offers, one per dataset.
- **Deployable model:** only features the app can collect: age, sex, lead time, weekday, earlier appointments, earlier no-show rate. Trained on Kaggle. The simulation and the staff flag use this one. The feature function lives in `ml/src/features.py` and the app imports the same function (no training/serving drift).

Models, in order: always-show baseline, earlier-no-show-rate rule baseline, logistic regression (class weights), random forest, gradient boosting (`HistGradientBoostingClassifier`; XGBoost only if time remains), then calibration (Platt or isotonic, chosen by Brier on grouped CV).

Features: lead time in days, appointment weekday, age, sex, SMS reminder (research only), health flags (research only), neighbourhood (research only; grouped or target-encoded inside CV folds only), and patient history.

**History rule:** earlier appointments and earlier no-show rate use only appointments whose date is before the current booking date (outcome known at booking time). This is stricter than "before the current appointment".

Evaluation:
- Lock a time-based holdout first (latest 20% of appointment dates, cut at a date boundary; for OpenML, hold out the last month). Never tune on it; score it once per final model.
- Tune with patient-grouped cross-validation on Kaggle (GroupKFold on PatientId); stratified CV on OpenML since it has no patient ID.
- Report AUC-ROC, average precision, sensitivity at fixed precision (30%, 40%, 50%) and Brier score, each with a 95% bootstrap interval (1,000 resamples of patients on the holdout). Never use accuracy as the headline; always-show scores about 80%.
- Imbalance: class weights and threshold tuning first. Resampling (RUS, SMOTE, NearMiss) is an ablation only, because it distorts probabilities. The simulation uses the unresampled, calibrated model.
- Leakage check: a validation AUC above about 0.85 is suspicious. Stop and investigate before reporting. Expect roughly 0.65 to 0.80.
- Check `SMS_received` carefully: reminders may be sent selectively, so compare with and without it and cross-tabulate against lead time. Describe the effect as an association.

Leakage and correctness tests: history never uses an outcome on or after the booking date; no patient in both a training fold and its validation fold; holdout dates after training dates; both loaders give no-show = 1 for the same hand-made rows; preprocessing fitted inside folds only; shuffled-label AUC near 0.5; one booking through the training and serving feature paths gives identical numbers.

Experiments, each writing to `results/`:

| ID | Question | Output |
|---|---|---|
| E1 | Dataset profile and cleaning | `cleaning_log.csv`, `dataset_summary.csv`, EDA figures |
| E2 | Baselines | `e2_baselines.csv` |
| E3 | Model comparison on Kaggle, all features | `e3_kaggle_models.csv`, ROC and PR figures |
| E4 | Feature-group ablation | `e4_ablation.csv` |
| E5 | Same ladder on OpenML | `e5_openml_models.csv` |
| E6 | Transfer on common features (age, sex, lead time, weekday), both directions; report the AUC drop | `e6_transfer.csv` |
| E7 | Imbalance handling comparison | `e7_imbalance.csv` |
| E8 | Calibration | `e8_calibration.csv`, reliability figure |
| E9 | Deployable model | `e9_deployable.csv`, `ml/artifacts/risk_model.joblib`, `model_card.json` |
| E10 | Threshold table (precision and sensitivity per threshold) | `e10_thresholds.csv` |

## Simulation plan (SimPy, discrete-event)

Single provider, one half-day session of `T` minutes with slots of `L` minutes (`N = T / L` slots). Patients arrive on time, served first come first served by slot time; a no-show uses no doctor time. Inputs:
- Consultation time: fit lognormal and gamma to Hangu service times (minutes), pick by AIC and QQ plots, and state the choice.
- Patient no-show probabilities: bootstrap patient profiles from the Kaggle holdout, scored by the calibrated deployable model; attendance is drawn from that probability.
- `T`, `L`, `N` are explicit parameters. Default `L` is Hangu's median service time rounded up to the next 5 minutes, computed by the script.

Policies: (P0) no overbooking, (P1) uniform: every k-th slot gets a second patient, k = 2, 3, 4, 6, (P2) risk-threshold double-booking, grid = fixed 0.3 and 0.5 plus the 50th, 70th, 80th, 90th, 95th percentiles of predicted risk. Stretch only if time remains: a simple reinforcement-learning policy, following Amalina & An (2026, arXiv preprint).

Draw patients, attendance and service times once per replication (a primary and an extra patient per slot) and let each policy choose which extras to use. At least 1,000 replications per setting, `base_seed + r` per replication, 95% intervals (paired differences against P0).

Metrics per policy: patients served, mean waiting time of attending patients, provider overtime, provider idle time, and share of sessions with both double-booked patients attending. **Report the trade-off table; do not collapse it to one weighted utility score**, since results depend on chosen weights (LaGanga & Lawrence, 2007).

Experiments: S1 sanity tests (all attend with service = L gives zero wait and overtime; nobody attends gives zero served and idle = T; hand-worked 3-patient case; same seed gives identical results), S2 policy trade-offs, S3 predicted risk shifted by -0.10, -0.05, 0, +0.05, +0.10 (overestimation hurts more, per Amalina & An), S4 base no-show rate scaled 0.5, 1, 1.5, S5 service-time variability scaled 0.5, 1, 1.5 at the same mean, **S6 value of prediction: compare P2 with P1 and with random scores at the same number of double-booked slots, and against an oracle**. S6 is the headline: more overbooking always serves more patients, so only an equal-rate comparison shows whether prediction adds anything. Outputs: `s2_policy_tradeoffs.csv`, `s3_probability_error.csv`, `s4_base_rate.csv`, `s5_service_variability.csv`, `s6_value_of_prediction.csv`.

**Mismatch to disclose in the report:** no-show behaviour comes from a Brazilian public-clinic dataset and consultation times from a Chinese private traditional-medicine clinic. The simulation demonstrates a method and its trade-offs, not clinical outcomes.

## Repository layout

```
app/        Flask app (blueprints: auth, patient, doctor, admin), services, risk.py, templates, static
db/         schema.sql, seed_synthetic.py
ml/         src/ (data, features, train, evaluate, calibrate), artifacts/, notebooks/
sim/        clinic.py, policies.py, experiments.py
data/       raw/ (gitignored), README.md with download steps
results/    tables, figures, manifest.json written by scripts
tests/      booking concurrency and constraint tests, leakage tests, simulation sanity tests, risk flag tests
docs/       PLAN.pdf, numbers_audit.csv, report drafts
```

## Build order (one milestone per session; commit when its checks pass)

| Milestone | Week | Done when |
|---|---|---|
| M0 Setup: layout, venv, requirements, .gitignore, README with MySQL steps | 1 | Tree exists, app connects to MySQL |
| M1 Database and booking core, tests first | 1 to 2 | Concurrency test passes on MySQL |
| M2 App screens and roles, seed script, role/slot/end-to-end tests | 2 to 4 | Log in as each role; book, cancel, reschedule |
| M3 Data pipeline: loaders, cleaning with logged counts, splits, E1 | 3 to 4 | `cleaning_log.csv` and `data/README.md` exist |
| M4 Research models E2 to E8, leakage tests first | 5 to 6 | Result files exist, leakage tests pass |
| M5 Deployable model E9, E10, model card, shared feature function | 7 | Model file, card, threshold table exist |
| M6 Simulation S1 to S6, sanity tests first | 7 to 8 | S1 to S6 files exist, sanity tests pass |
| M7 Risk flag in the app: `app/risk.py`, `risk_scores`, staff badges (High and Medium from the thresholds on the model card) | 9 | Flag shows; booking works without the model |
| M8 Report, figures from `results/`, numbers audit, demo | 8 to 10 | Every report number traces to a results file |

Gates: booking proven (concurrency test on MySQL), model frozen (E9 and E10 exist), results frozen (no experiment rerun after).

If time runs short, cut in this order: reinforcement-learning policy, standby list, XGBoost, resampling ablation (E7). Never cut the booking constraint and its tests, E3, E8, E9, S2, S6, or the numbers audit.

## Report notes

- Chapters: 1 Introduction, 2 Literature Survey, 3 Limitations of the Existing Works, 4 Problem Statement, 5 Objectives, 6 Proposed Design, 7 Implementation and Testing, 8 Results, 9 Limitations and Future Work, 10 Conclusion, 11 References. Details in `docs/PLAN.pdf` section 8.
- Literature references: see the CAMS Report: Revised Sections 2 to 4 document (23 corrected entries).
- Amalina & An (2026) is an arXiv preprint; label it as such.
- Open items: confirm Fan et al. pages 469-490, full author lists for Deina et al. and Zhou et al.
- The model uses age and sex; say so under limitations. The flag is advisory and no patient is refused a slot because of it.
