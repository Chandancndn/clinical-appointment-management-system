"""E1, the dataset profile: runs the whole pipeline on small synthetic raw files and checks the outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.src import data, e1_profile
from test_data_loaders import k_row, o_row, h_row, write_hangu, write_kaggle, write_openml


def build_raw(tmp_path, *, holdout_all_missed=False):
    """Small synthetic raw files in the real formats. 25 weekday dates, repeat patients."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2016-05-02", periods=25)
    kaggle_rows = []
    for i in range(500):
        day = dates[i % len(dates)]
        scheduled = day - pd.Timedelta(days=int(rng.choice([0, 1, 3, 9, 20, 40])))
        late = day >= dates[-5]  # the last 5 dates become the holdout
        missed = (1 if late else 0) if holdout_all_missed else int(rng.random() < 0.2)
        kaggle_rows.append(k_row(
            PatientId=str(10_000_000_000 + int(rng.integers(0, 120))), Age=int(rng.integers(0, 90)),
            Gender="F" if rng.random() < 0.6 else "M", ScheduledDay=scheduled.strftime("%Y-%m-%dT10:30:00Z"),
            AppointmentDay=day.strftime("%Y-%m-%dT00:00:00Z"), SMS_received=int(rng.random() < 0.3),
            **{"No-show": "Yes" if missed else "No"}))
    kaggle_rows.append(k_row(Age=-1))  # a row the cleaner must drop
    openml_rows = [o_row(edad=int(rng.integers(0, 90)), sexo=int(rng.integers(1, 3)),
                         reserva_mes_d=int(1 + i % 4), reserva_dia_d=int(1 + rng.integers(0, 6)),
                         reserva_hora_d=int(rng.integers(8, 19)), latencia=int(rng.choice([0, 2, 6, 12, 25, 50])),
                         canal=int(rng.integers(1, 4)), especialidad=int(rng.integers(1, 20)),
                         show=int(rng.random() > 0.2)) for i in range(300)]
    hangu_rows = [h_row(Session=1 + i // 6, StartTime=f"8:{i % 60:02d}:00", ServTime=int(rng.integers(200, 2000)))
                  for i in range(80)]
    return (write_kaggle(tmp_path / "k.csv", kaggle_rows), write_openml(tmp_path / "o.arff", openml_rows),
            write_hangu(tmp_path / "h.csv", hangu_rows))


@pytest.fixture
def run_e1(tmp_path):
    def run(**build_kwargs):
        kaggle, openml, hangu = build_raw(tmp_path, **build_kwargs)
        results = tmp_path / "results"
        summary = e1_profile.run(kaggle_path=kaggle, openml_path=openml, hangu_path=hangu,
                                 results_dir=results, repo_root=tmp_path)
        return results, summary

    return run


def metric(summary_table: pd.DataFrame, dataset: str, name: str):
    row = summary_table[(summary_table["dataset"] == dataset) & (summary_table["metric"] == name)]
    assert len(row) == 1, f"{dataset}/{name}: {len(row)} rows"
    return row["value"].iloc[0]


def test_e1_writes_the_log_the_summary_the_rates_and_the_figures(run_e1):
    results, _ = run_e1()
    for name in ("cleaning_log.csv", "dataset_summary.csv", "e1_eda_rates.csv", "manifest.json"):
        assert (results / name).stat().st_size > 0, name
    figures = sorted(p.name for p in (results / "figures").glob("*.png"))
    assert figures == ["e1_noshow_by_age_band.png", "e1_noshow_by_lead_time.png",
                       "e1_noshow_by_weekday.png", "e1_overall_noshow_rate.png"]
    for png in (results / "figures").glob("*.png"):
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_cleaning_log_covers_all_three_datasets(run_e1):
    results, _ = run_e1()
    log = pd.read_csv(results / "cleaning_log.csv")
    assert set(log["dataset"]) == {"kaggle", "openml", "hangu"}
    dropped_age = log.query("dataset == 'kaggle' and rule == 'drop_impossible_age'")["rows_changed"].iloc[0]
    assert dropped_age == 1  # the planted age = -1 row


def test_the_summary_numbers_are_computed_from_the_data(run_e1, tmp_path):
    results, _ = run_e1()
    table = pd.read_csv(results / "dataset_summary.csv")
    assert list(table.columns) == ["dataset", "metric", "value", "note"]
    raw = data.load_kaggle(tmp_path / "k.csv")
    clean = data.clean_kaggle(raw)
    assert int(metric(table, "kaggle", "rows_raw")) == len(raw)
    assert int(metric(table, "kaggle", "rows_clean")) == len(clean)
    assert float(metric(table, "kaggle", "no_show_rate_clean")) == pytest.approx(clean["no_show"].mean())
    assert int(metric(table, "kaggle", "unique_patients_clean")) == clean["patient_id"].nunique()
    assert int(metric(table, "openml", "rows_raw")) == 300
    assert int(metric(table, "hangu", "rows_clean")) == 80


def test_the_summary_describes_the_locked_holdout_and_proves_the_folds_are_patient_disjoint(run_e1):
    results, _ = run_e1()
    table = pd.read_csv(results / "dataset_summary.csv")
    assert int(metric(table, "kaggle", "n_train_rows")) + int(metric(table, "kaggle", "n_holdout_rows")) == \
        int(metric(table, "kaggle", "rows_clean"))
    assert pd.Timestamp(metric(table, "kaggle", "holdout_first_date")) > pd.Timestamp(metric(table, "kaggle", "train_last_date"))
    assert int(metric(table, "kaggle", "cv_patients_in_both_sides")) == 0
    assert int(metric(table, "openml", "holdout_month")) == 4
    assert int(metric(table, "openml", "n_train_rows")) + int(metric(table, "openml", "n_holdout_rows")) == \
        int(metric(table, "openml", "rows_clean"))


def test_the_eda_rates_use_only_the_training_part_never_the_holdout(run_e1):
    """Plant every no-show in the holdout dates: a profile that peeked at them would show high rates."""
    results, _ = run_e1(holdout_all_missed=True)
    rates = pd.read_csv(results / "e1_eda_rates.csv")
    kaggle = rates[(rates["dataset"] == "kaggle") & (rates["variable"] != "overall")]
    assert kaggle["no_show_n"].sum() == 0 and kaggle["n"].sum() > 0
    summary = pd.read_csv(results / "dataset_summary.csv")
    assert float(metric(summary, "kaggle", "holdout_no_show_rate")) == 1.0  # the holdout is described, separately


def test_the_eda_rate_table_is_well_formed(run_e1):
    results, _ = run_e1()
    rates = pd.read_csv(results / "e1_eda_rates.csv")
    assert list(rates.columns) == ["dataset", "variable", "group", "n", "no_show_n", "rate", "ci_low", "ci_high"]
    assert set(rates["variable"]) == {"lead_time_band", "weekday", "age_band", "overall"}
    assert (rates["ci_low"] <= rates["rate"]).all() and (rates["rate"] <= rates["ci_high"]).all()
    assert rates["rate"].between(0, 1).all()
    for (dataset, variable), group in rates[rates["variable"] != "overall"].groupby(["dataset", "variable"]):
        assert group["n"].sum() == rates.query("dataset == @dataset and variable == 'lead_time_band'")["n"].sum()


def test_every_output_has_a_manifest_entry_with_the_raw_data_hashes(run_e1):
    results, _ = run_e1()
    stored = json.loads((results / "manifest.json").read_text())
    expected = {f"results/{n}" for n in ("cleaning_log.csv", "dataset_summary.csv", "e1_eda_rates.csv")} | \
               {f"results/figures/{n}" for n in ("e1_noshow_by_age_band.png", "e1_noshow_by_lead_time.png",
                                                  "e1_noshow_by_weekday.png", "e1_overall_noshow_rate.png")}
    assert set(stored) == expected
    for entry in stored.values():
        assert entry["seed"] == data.DEFAULT_SEED and entry["script"].endswith("e1_profile.py")
        assert len(entry["data_files"]) == 3 and all(len(h) == 64 for h in entry["data_files"].values())


def test_rerunning_gives_identical_tables(run_e1):
    results, _ = run_e1()
    first = {n: (results / n).read_bytes() for n in ("cleaning_log.csv", "dataset_summary.csv", "e1_eda_rates.csv")}
    run_e1()
    assert first == {n: (results / n).read_bytes() for n in first}


def test_wilson_interval():
    low, high = e1_profile.wilson_interval(0, 10)
    assert low == 0.0 and 0.27 < high < 0.29
    low, high = e1_profile.wilson_interval(5, 10)
    assert low == pytest.approx(1 - high) and 0.23 < low < 0.25
    low, high = e1_profile.wilson_interval(2000, 10000)
    assert 0.19 < low < 0.20 < high < 0.21
    assert all(np.isnan(v) for v in e1_profile.wilson_interval(0, 0))
