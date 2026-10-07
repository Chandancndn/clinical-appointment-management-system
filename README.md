# CAMS: Clinic Appointment Management System

V-semester CSE mini project, SIT Tumakuru. Flask + SQLAlchemy + MySQL, with a no-show
prediction model (scikit-learn) and a booking-policy simulation (SimPy).

- Rules for contributors and AI assistants: [CLAUDE.md](CLAUDE.md)
- Design and build order: the "CAMS Master Plan" document (`docs/PLAN.pdf`)

The central guarantee is that the same doctor, date and time can never be booked twice.
The **database** enforces it (a unique key on `slots` and a unique generated column on
`bookings`), not a Python check. See `db/schema.sql`.

## Setup

Requires Python 3.9 or newer and MySQL 5.7 or newer (generated columns need 5.7+; 8.x is
fine). Tested with Python 3.14.6 and MySQL 26.7.0 (Homebrew, October 2026).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env, see below
```

## MySQL setup

1. Install and start MySQL. On macOS with Homebrew:

   ```bash
   brew install mysql
   brew services start mysql
   mysql_secure_installation      # optional but recommended
   ```

2. Create the databases and a user. Open `mysql -u root -p` and run:

   ```sql
   CREATE DATABASE cams      CHARACTER SET utf8mb4;
   CREATE DATABASE cams_test CHARACTER SET utf8mb4;
   CREATE USER 'cams'@'localhost' IDENTIFIED BY 'choose-a-password';
   GRANT ALL PRIVILEGES ON cams.*      TO 'cams'@'localhost';
   GRANT ALL PRIVILEGES ON cams_test.* TO 'cams'@'localhost';
   ```

   `cams_test` is separate on purpose: the tests drop and recreate every table in it.
   The test runner refuses to use a database whose name does not end in `_test`.

3. Put the connection details in `.env` (copy of `.env.example`):

   ```
   DATABASE_URL=mysql+pymysql://cams:choose-a-password@localhost:3306/cams?charset=utf8mb4
   TEST_DATABASE_URL=mysql+pymysql://cams:choose-a-password@localhost:3306/cams_test?charset=utf8mb4
   SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_hex(32))">
   ```

   If the password contains `@`, `:`, `/` or `#`, URL-encode it.

4. Check the connection and create the tables:

   ```bash
   python -m db.check_connection    # prints the server version and whether generated columns work
   python -m db.init_db             # applies db/schema.sql to the database in DATABASE_URL
   ```

## Run the app

```bash
python -m db.seed_synthetic      # synthetic demo data only; prints the demo logins
flask --app app run --port 5001  # then open http://127.0.0.1:5001
```

(Port 5001 because macOS uses 5000 for AirPlay. Add `--debug` for auto-reload on your own machine.)

`db.seed_synthetic` creates 1 admin, 5 doctors and 40 patients with obviously fake names, slots for
the next and previous 14 weekdays, some future bookings (a few cancelled) and past appointments with
outcomes. It never reads `data/raw/` or any dataset. Every demo account shares one password:
`SEED_PASSWORD` from `.env` (or a random one, printed when the script runs). Demo accounts:

| Role | Email |
|---|---|
| Admin | `admin@cams-demo.test` |
| Doctor | `dr.meera@cams-demo.test` (also `dr.arjun`, `dr.kavya`, `dr.rohan`, `dr.sana`) |
| Patient | `asha.demo01@cams-demo.test` (also `<name>.demo02` ... `.demo40`) |

Re-seeding needs `python -m db.seed_synthetic --reset`, which deletes every row first.

## Risk flag (advisory)

Doctors and admins see a **Low / Medium / High** badge beside each appointment, with the note "advisory only". It
comes from the saved deployable model (`ml/artifacts/risk_model.joblib`), uses the Medium and High thresholds on
`ml/artifacts/model_card.json` (never typed into the app), and is computed by the same `features.deployable_features()`
that trained the model.

- Booking never reads it. `app/services.py` does not import `app/risk.py`; a booking is scored **after** it has been
  committed, inside a try/except, and scores live in their own table (`risk_scores`), not on `bookings`.
- If the model file or card is missing, does not match its hash, or scoring raises, the badges simply do not appear and
  booking, cancelling and rescheduling behave exactly as before.
- Patients never receive risk data, and no patient is refused or moved because of a flag.

To see the badges: `python -m db.init_db` (adds `risk_scores` if missing), `python -m db.seed_synthetic --reset`
(scores the demo bookings and prints the Low/Medium/High mix), start the app, and log in as the doctor
(`dr.meera@cams-demo.test`, **My schedule**, a day with appointments) or the admin (**Overview** and **All bookings**).
Log in as a patient to confirm there is nothing to see. Re-running E9 and E10 changes the model version, and the next
staff page view re-scores what it shows.

## Tests

```bash
pytest                       # SQLite only (default)
pytest --engine mysql        # the same tests on MySQL (needs TEST_DATABASE_URL, see above)
pytest --engine both         # both engines
```

The booking tests must pass on MySQL, not only SQLite, because the double-booking
constraint has to be proven on the engine the project actually uses.

## Research experiments (ML)

Needs the raw datasets in `data/raw/` (see `data/README.md`). Every script writes its tables and figures to
`results/` and records them in `results/manifest.json` (seed, raw-data hashes, date, git commit). Run them in this
order, one at a time (they share the manifest, and E4, E7 and E8 read the model E3 selected):

```bash
python -m ml.src.e1_profile          # E1  dataset profile and cleaning log
python -m ml.src.e2_baselines        # E2  always-show and the earlier-no-show-rate rule
python -m ml.src.e3_kaggle_models    # E3  logistic regression, random forest, gradient boosting; ROC/PR figures
python -m ml.src.e4_ablation         # E4  feature-group ablation and the SMS x lead-time check
python -m ml.src.e5_openml_models    # E5  the same ladder on OpenML
python -m ml.src.e6_transfer         # E6  train on one dataset, test on the other
python -m ml.src.e7_imbalance        # E7  class weights, thresholds, RUS, SMOTE, NearMiss (ablation only)
python -m ml.src.e8_calibration      # E8  Platt vs isotonic, reliability figure
python -m ml.src.e9_deployable       # E9  the six-feature deployable model -> ml/artifacts/risk_model.joblib + model_card.json
python -m ml.src.e10_thresholds      # E10 threshold table; chooses Medium/High and writes them into the model card
```

The deployable model uses only what the app can collect at booking time. Both the app and training compute its inputs
with the single function `features.deployable_features()` (a test pushes one booking through both paths).

The holdout (latest 20% of Kaggle appointment dates; the last OpenML month) is scored once per final model, and a
run stops if any AUC exceeds 0.85 (suspected leakage). Hyper-parameters are tuned by patient-grouped
cross-validation on the training part only.

## Simulation (booking policies)

```bash
python -m sim.experiments     # S2 to S6 plus the consultation-time fit; needs the raw data and ml/artifacts/risk_model.joblib
```

One provider, one half-day session (T minutes from the seeded app slots, slots of L minutes where L is Hangu's median
consultation time rounded up to the next 5), at least 1,000 replications with seed `base + r`, and common random
numbers: every policy sees the same patients, attendance and consultation times. It uses the saved deployable model and
never retrains. Tables and figures go to `results/` (`s2_policy_tradeoffs.csv` ... `s6_value_of_prediction.csv`) with
manifest entries that record T, L, N and the replication count. The sanity tests (S1) are `tests/test_sim.py`.

## Repository layout

```
app/        Flask app: blueprints/ (auth, patient, doctor, admin), services.py (bookings),
            accounts.py, scheduling.py, queries.py (read-only), security.py, risk.py (advisory flag), templates/, static/
db/         schema.sql (MySQL), init_db.py, check_connection.py, seed_synthetic.py
ml/         src/, artifacts/, notebooks/      (M3 onwards)
sim/        simulation                        (M6)
data/       README.md with download steps; raw files go in data/raw/ (gitignored)
results/    tables, figures, manifest.json written by scripts
tests/      booking, leakage, simulation and risk-flag tests
docs/       PLAN.pdf, numbers_audit.csv, report drafts
```

Raw datasets are never committed. Download steps will be in `data/README.md` (milestone M3).
