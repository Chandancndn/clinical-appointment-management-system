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

## Tests

```bash
pytest                       # SQLite only (default)
pytest --engine mysql        # the same tests on MySQL (needs TEST_DATABASE_URL, see above)
pytest --engine both         # both engines
```

The booking tests must pass on MySQL, not only SQLite, because the double-booking
constraint has to be proven on the engine the project actually uses.

## Repository layout

```
app/        Flask app: blueprints/ (auth, patient, doctor, admin), services.py (bookings),
            accounts.py, scheduling.py, queries.py (read-only), security.py, templates/, static/
db/         schema.sql (MySQL), init_db.py, check_connection.py, seed_synthetic.py
ml/         src/, artifacts/, notebooks/      (M3 onwards)
sim/        simulation                        (M6)
data/       README.md with download steps; raw files go in data/raw/ (gitignored)
results/    tables, figures, manifest.json written by scripts
tests/      booking, leakage, simulation and risk-flag tests
docs/       PLAN.pdf, numbers_audit.csv, report drafts
```

Raw datasets are never committed. Download steps will be in `data/README.md` (milestone M3).
