# CAMS demo script (about five minutes)

The order follows the story of the project: the app first, then the proof that it cannot double-book, then the advisory risk flag, then the results. Say the "Say" lines in your own words; they are the point of each step. No result number is quoted in this script, so there is nothing here to go out of date: when you show a result, read it off the figure or table.

## Before you start (two minutes, once)

1. Start MySQL, then from the repository: `source venv/bin/activate`.
2. Re-seed so the demo dates are around today: `python -m db.seed_synthetic --reset`. It prints the demo logins and the Low, Medium and High mix of the risk flag. The shared password for every seeded account is `SEED_PASSWORD` in `.env`.
3. Start the app: `flask --app app run --port 5001`, and open http://127.0.0.1:5001.
4. Open **two browser windows** side by side: a normal one and a private one, so two people can be logged in at once.
5. Have these ready in tabs: `docs/report_assets/figures/f16_s6_paired_differences.png`, `f11_s2_tradeoffs.png`, `f10_e10_thresholds.png`, and a terminal in the repository.

Accounts (all in the seeded database):

| Role | Email |
|---|---|
| Admin | `admin@cams-demo.test` |
| Doctor | `dr.meera@cams-demo.test` |
| Patient | `asha.demo01@cams-demo.test` |

## The five minutes

**0:00 Introduce (30 seconds).** Show the login page.
*Say:* a clinic appointment system for patients, doctors and admins. Two rules drive the design: the same doctor, date and time can never be booked twice, enforced by the database; and the app never overbooks. A model and a simulation study overbooking separately, and the app only shows staff an advisory flag.

**0:30 Admin adds a doctor (45 seconds).** Log in as the admin (normal window). Click **Add doctor**, enter a name, an email such as `dr.demo@cams-demo.test`, a password you will remember, a specialization and an appointment length. You land on **Generate slots**: choose a date range of a few days, a morning window, tick *skip weekends*, and submit.
*Say:* roles are checked on every route, passwords are hashed, every form carries a CSRF token. Slots are generated, not typed; re-running it never duplicates one, because of the unique key on doctor, date and time.

**1:15 A patient registers and books (45 seconds).** First set the trap for the next step: in the normal window log out of the admin, log in as `asha.demo01@cams-demo.test`, open **Doctors**, click **View free slots** for the new doctor, and leave the first day's slot list open without clicking anything. Now, in the private window, click **Register**: name, email such as `demo.patient@cams-demo.test`, password, date of birth, sex; the app sends you to the login page, so log in. Open **Doctors**, click **View free slots** for the new doctor, and click the time of the first free slot (the reason box is optional). Open **My appointments** to show it.
*Say:* the date of birth and sex are what the risk model needs; they are never shown to other patients.

**2:00 A second patient tries the same slot (45 seconds).** This is the proof of rule one. Go back to the normal window: Asha's page was loaded before the booking, so it still lists that slot as free. Click that time.
*Expected:* Asha sees "That slot was just taken. The list below is up to date; please pick another." (HTTP 409), and the refreshed list no longer offers the slot.
*Say:* the app does not check and then insert; it just inserts, and the database refuses the second booking. In the terminal, optionally run `pytest tests/test_booking.py --engine mysql -q` to show the concurrent-request tests pass on real MySQL. The number of tests run is in `docs/report_assets/tables/t21_test_summary.md`.

**2:45 The doctor marks outcomes (45 seconds).** Log in as `dr.meera@cams-demo.test`. Open **My schedule**, step back to an earlier day with appointments (the date box or **Previous day**), and use **Mark no-show** or **Mark completed** on one of them.
*Say:* outcomes can only be set once the appointment time has started; a cancelled booking frees the slot, but completed and no-show ones keep it. These outcomes are the history the model's features use. The booking you just made in the private window is in the future, so its buttons appear only when its time comes.

**3:30 The risk badges (45 seconds).** Still as the doctor, go to a day with several bookings: each row has a **Low**, **Medium** or **High** badge and the note "Risk badges are advisory only". Then log in as the admin and show **Overview** and **All bookings**. Finally open a patient's **My appointments** page: no risk anywhere.
*Say:* the flag comes from the deployable model with the six features the app can collect, using the cut-offs on the model card. It is computed after a booking is saved, inside a try/except. Booking never reads it: if the model file is missing or scoring fails, booking works exactly as before (`tests/test_risk_flag.py` proves both). Nobody is refused or moved because of a flag.

**4:15 Models and simulation (45 seconds).** Log in as the admin and open **Models and simulation** (and the *Risk flag status* panel at the bottom of **Overview**): the model card, the three bands with their measured no-show rates, the policy trade-off table and the figures, all read from the saved result files. Scroll the page to the figures below.
*Say:* the model is a gradient-boosted tree on six inputs the app can collect; the simulation is one doctor and one half-day session, with consultation times from one clinic and no-show risk from another, so it shows a method and its trade-offs, not clinical outcomes. The app itself never overbooks.

**5:00 Results files (optional).** Switch to the figures.
1. `f16_s6_paired_differences.png` is the headline of the simulation: at the same number of double-booked slots, does choosing by predicted risk beat spacing them evenly or choosing at random? Read the answer straight off the plot: the gain is modest and not uniform, and an oracle shows how much more perfect knowledge would give.
2. `f11_s2_tradeoffs.png` shows that more patients served always costs more waiting and overtime, so no single score is given.
3. `f10_e10_thresholds.png` shows what each Low, Medium and High threshold catches.
4. In the terminal run `python -m scripts.check_numbers --fresh`: every number in the report assets and summary is re-read from `results/` and matches.
*Say:* the data is public (Brazil for no-shows, China for consultation times), the clinic behaviour in the simulation is assumed, and the flag is advisory. This shows a method and its trade-offs, not clinical outcomes.

## If something goes wrong

- *The new patient's booking is refused as "in the past":* the seed was run on an earlier day. Re-run the seed with `--reset`.
- *No badges appear:* the model file is missing or does not match its model card, which is the designed behaviour. Check `ml/artifacts/risk_model.joblib` and `model_card.json`, and re-run `python -m db.seed_synthetic --reset`.
- *Two windows share one login:* use a private window for the second user.
- *MySQL is not running:* start it (`brew services start mysql`) and run `python -m db.check_connection`.

## Screenshots to capture for Chapter 7

Take these from the running app (PLAN section 8 asks for each role's pages): the login page; the patient's doctor list, slot page and **My appointments**; the "slot was just taken" message; the doctor's **My schedule** with badges; the admin's **Overview** and **All bookings** with badges; the **Add doctor** and **Generate slots** forms.
