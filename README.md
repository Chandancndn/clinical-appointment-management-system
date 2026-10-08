# CAMS: Clinic Appointment Management System

V-semester CSE mini project, SIT Tumakuru. Flask + SQLAlchemy + MySQL, with a no-show
prediction model (scikit-learn) and a booking-policy simulation (SimPy).

- **Live demo:** https://clinical-appointment-management-sys-one.vercel.app (Vercel, with a MySQL 8 database on Aiven; synthetic data only)
- Rules for contributors and AI assistants: [CLAUDE.md](CLAUDE.md)
- Design and build order: the "CAMS Master Plan" document (`docs/PLAN.pdf`)

The central guarantee is that the same doctor, date and time can never be booked twice.
The **database** enforces it (a unique key on `slots` and a unique generated column on
`bookings`), not a Python check. See `db/schema.sql`.

## Setup

Requires Python 3.9 or newer and MySQL 5.7 or newer (generated columns need 5.7+; 8.x is
fine). Developed and tested with Python 3.14.6 and MySQL 26.7.0 (Homebrew, October 2026). A fresh clone was also set up from these steps on the macOS system Python 3.9.6, with the older library versions pip chooses for it (for example pandas 2.3 and scikit-learn 1.6): the whole test suite passed on both engines, apart from the tests that need the raw datasets, which are skipped.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-research.txt   # the web app, the experiments, the simulation and the tests
cp .env.example .env                       # then edit .env, see below
```

`requirements.txt` alone is what the web app needs (and what Vercel installs, see "Deploy to Vercel"); `requirements-research.txt` adds the experiment, simulation, plotting and test tools.

**Saved model and scikit-learn.** `ml/artifacts/risk_model.joblib` was written with the library versions recorded in `ml/artifacts/model_card.json` (`environment`). `requirements.txt` pins scikit-learn, numpy and joblib to those versions wherever they can be installed (Python 3.11 and later). On Python 3.9 or 3.10 the app still works, but scikit-learn warns on load, a few bookings near a threshold may get a different badge, and the admin overview says which versions differ.

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
   python -m db.init_db             # applies db/schema.sql to the database in DATABASE_URL; also upgrades an older database
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

## What the app does

| Role | Can |
|---|---|
| Patient | Register, log in, browse doctors and free slots, book, cancel, reschedule, join the standby list for a day that is full, change own password |
| Doctor | See own schedule with patient names and the advisory risk badge, create slots for a day, close and reopen free slots (leave), see who is on standby for a day, cancel an appointment, mark past appointments completed or no-show, change own password |
| Admin | Add doctors, generate slots, close and reopen any doctor's free slots, see and cancel any booking, mark outcomes, see the risk flag status and the **Models and simulation** page |

Rules the app enforces: a doctor's slot can never be booked twice (the database decides); a patient cannot hold two appointments whose times overlap (checked on the page, best effort); a started appointment cannot be cancelled or moved; a patient never sees another patient's booking (404) or any risk information. The app never overbooks.

**Leave.** Closing a slot inserts a row with status `closed` in the doctor's name; it holds the slot through the same unique key that forbids double-booking, so a patient booking and a doctor closing the same slot are decided by the database, and exactly one wins. Closing only affects free slots that have not started and never cancels an appointment; reopening only touches closed slots.

**Standby.** A patient can join the standby list for a doctor on a day with no free slot. Standby reserves nothing and gives no priority: when a slot on that day is free again the patient sees it on their own pages and books it the normal way, first come first served. Staff see who is waiting for a day.

Safeguards: passwords are hashed; every POST carries a CSRF token; every route checks the role; five wrong passwords for one email from one address (twenty from one address) block further tries for fifteen minutes, also for emails that do not exist, and guessing the current password on the account page is limited the same way. The counters live in memory, per process, and use the address Flask sees, so behind a reverse proxy or several workers they bound an attacker per worker rather than globally. Set `SESSION_COOKIE_SECURE=true` in `.env` when serving over HTTPS. Ids and dates in URLs and forms are range-checked, so odd input gives a 404 or a message, never a server error (`tests/test_robustness.py` sends junk to every route as every role).

The tables become cards on a phone-width screen.

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
Log in as a patient to confirm there is nothing to see. The admin **Overview** also has a *Risk flag status* panel (model version, thresholds, whether the flag is on and why not, and the observed no-show rate per band on recorded outcomes), and **Models and simulation** shows the model card, the three bands and the booking-policy simulation with its figures, all read from the saved result files. Re-running E9 and E10 changes the model version, and the next
staff page view re-scores what it shows.

## Deploy to Vercel

On Vercel the app is one Python function (`index.py`, which builds the same app as `flask --app app run`) and the database
is a **hosted MySQL**. Use a real MySQL 8 service: the no-double-booking rule rests on MySQL's unique generated column,
and Vercel has no database that offers it. Vercel installs `requirements.txt` only (the research tools stay out, which
keeps the bundle well under Vercel's 500 MB limit) and uses the Python version in `.python-version`.

**1. A database.** Create a MySQL 8 database and note the host, port, user, password and database name. If the provider
requires TLS, save its CA certificate (a public file) as `certs/ca.pem`. The URL looks like:

```
mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME?charset=utf8mb4&ssl_ca=certs/ca.pem
```

(URL-encode special characters in the password; drop `&ssl_ca=...` if the provider does not use TLS.)

**2. Prepare that database from your laptop, once.** Never from Vercel. The first line prints the server version and
whether generated columns work; the last one fills it with synthetic demo data (obviously fake people, no dataset is read):

```bash
export DATABASE_URL='mysql+pymysql://...'      # the URL from step 1
python -m db.check_connection
python -m db.init_db
SEED_PASSWORD='pick-a-demo-password' python -m db.seed_synthetic
```

To prove the booking rule on that host before you rely on it, create a second, empty database whose name ends in
`_test` on the same server and run the booking tests there: `TEST_DATABASE_URL='...' pytest --engine mysql`.

**3. The Vercel project.** Import the repository, leave the framework as **Flask** (set it by hand if it was not detected),
and add these environment variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | the URL from step 1 |
| `SECRET_KEY` | a new random value: `python -c "import secrets; print(secrets.token_hex(32))"` (not the one in your local `.env`) |
| `CLINIC_TIMEZONE` | `Asia/Kolkata` (or your clinic's zone). Vercel's servers run on UTC; without this, "has this slot started?" would be wrong by hours |

Do not add `SEED_PASSWORD` there. `index.py` already turns on secure cookies, tells the app to trust Vercel's proxy
(so the login throttle sees each visitor's own address) and makes database connections survive a frozen instance.

**Good to know.**
- Vercel serves static files from `public/`, not from `app/static/`. After changing anything in `app/static/` run
  `python -m scripts.sync_public` and commit `public/`; a test fails if the two differ.
- The login throttle counts in memory, per serverless instance, so on Vercel it slows guessing rather than stopping it.
- The demo slots are relative to the day you seeded. Before showing the app on a later day, re-run step 2's last line
  with `--reset` added.
- The risk flag ships with the app (the saved model plus scikit-learn, pandas and numpy). If the bundle ever hits the
  size limit, delete those lines from `requirements.txt`: the app still runs, the admin overview says the flag is off,
  and booking is unaffected by design.
- `tests/test_vercel_ready.py` checks the entry point, the static mirror, the proxy handling and that secrets and raw
  data are never uploaded.
- The site is live at https://clinical-appointment-management-sys-one.vercel.app, built from the `deploy-ready` branch. The booking, schema and leave tests also pass
  against the hosted MySQL (Aiven, server 8.4.8).

**Lessons from the first deploy.**
- Vercel builds the Production Branch, which defaults to `main`. Build the branch that has `index.py` (Project, Settings,
  Environments, Production, Branch Tracking). A plain Redeploy reuses the old commit; push a new commit to build the new branch.
- `DATABASE_URL` must begin `mysql+pymysql://`. A lost first letter crashes the app at start-up with
  "Can't load plugin" in the runtime logs.
- Use a MySQL service, not PostgreSQL: the schema and the double-booking rule need MySQL's generated column.
- Keep database URLs and passwords in a git-ignored file (`.env.*` is ignored) and in Vercel's environment variables, never in chat or in git.

## Tests

```bash
pytest                       # SQLite only (default)
pytest --engine mysql        # the same tests on MySQL (needs TEST_DATABASE_URL, see above)
pytest --engine both         # both engines
```

The booking tests must pass on MySQL, not only SQLite, because the double-booking
constraint has to be proven on the engine the project actually uses.

A few tests (loaders, a leakage check and the simulation experiments) need the raw datasets in `data/raw/` and are skipped without them; the skip message names the file. `python -m scripts.run_test_summary` runs everything on both engines and writes the summary the report quotes.

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

```bash
python -m sim.rl              # S7: a policy learned by reinforcement learning, one per stated cost of delay (about a minute)
```

S7 (`sim/rl.py`) learns where to double-book instead of fixing a rule by hand: REINFORCE on simulated sessions, with a policy that
sees the same predicted risk as P2 plus its own earlier choices. Learning needs one number to maximise, so three stated costs of
waiting and overtime are fixed in the code and a policy is learned for each. The results (`s7_*.csv`) still report every metric for
each learned policy, compare it with every hand-made rule under the same costs and at the same number of double-booked slots, and
try it with a biased risk estimate. Training, validation and evaluation sessions use disjoint seeds. It rewrites nothing from S2 to S6.

## Report assets and the numbers audit

Every number in the report comes from a file in `results/` (CLAUDE.md hard rule 3). Three scripts keep it that way; none of them trains a model or runs a simulation:

```bash
python -m scripts.run_test_summary       # whole test suite on SQLite and MySQL -> results/test_summary.csv and .txt
python -m scripts.make_report_assets     # docs/report_assets/ (tables as CSV and Markdown, figures as PNG), docs/RESULTS_SUMMARY.md, docs/numbers_audit.csv
python -m scripts.check_numbers --fresh  # fails if a quoted number no longer matches its results file, or docs/ was edited by hand
```

- `docs/report_assets/INDEX.md` lists every table and figure, the experiment it belongs to, its source file and the report chapter.
- `docs/RESULTS_SUMMARY.md` is written from `scripts/results_summary.md.j2`: numbers can only enter through `q()`, `qci()` and `lit()`, so a bare number makes the build fail, and a sentence that the results stop supporting also makes it fail. To change the wording, edit the template and rebuild; do not edit the summary.
- `docs/numbers_audit.csv` has one row per quoted number: the number as written, the results file, the row selector and column it comes from, and the document that shows it.
- `docs/DEMO_SCRIPT.md` is the five-minute demo order.

The raw datasets are not in the repository (CLAUDE.md hard rule 5), so the experiments cannot be re-run on a new machine until the files are downloaded as `data/README.md` describes. Everything above works without them, from the committed `results/`.

## Repository layout

```
app/        Flask app: blueprints/ (auth, account, patient, doctor, admin), services.py (bookings), standby.py,
            accounts.py, scheduling.py, queries.py (read-only), security.py, risk.py (advisory flag), research.py,
            clock.py, templates/, static/
db/         schema.sql (MySQL), migrations.py, init_db.py, check_connection.py, seed_synthetic.py
ml/         src/ (data, features, training and the E1 to E10 experiments), artifacts/ (saved model and card)
sim/        the booking-policy simulation (S1 to S6) and the learned policy (S7, rl.py)
data/       README.md with download steps; raw files go in data/raw/ (gitignored)
results/    tables, figures, manifest.json written by scripts
scripts/    make_report_assets.py, check_numbers.py, run_test_summary.py, sync_public.py, results_summary.md.j2
tests/      booking, leakage, simulation, risk-flag, report-numbers and deployment tests
docs/       PLAN.pdf, RESULTS_SUMMARY.md, DEMO_SCRIPT.md, numbers_audit.csv, report_assets/, ENTIRE_PROJECT_REPORT.html (the whole project explained for readers new to it)
index.py    Vercel entry point (not used locally)
vercel.json, .vercelignore, .python-version, public/    Vercel settings; public/static is a copy of app/static
requirements.txt (the web app), requirements-research.txt (adds the experiment, simulation and test tools)
certs/ca.pem    the hosted database's public CA certificate (used by `ssl_ca=certs/ca.pem` in DATABASE_URL)
```

Raw datasets are never committed. Download steps are in `data/README.md`.
