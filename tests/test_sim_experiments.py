"""S2 to S6 on small synthetic inputs: structure, consistency between experiments, and the equal-count design of S6."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from ml.src import data
from sim import experiments, inputs

T, L = 240, 15
N = T // L
N_REPS = 300


@pytest.fixture(scope="module")
def sim_inputs():
    service = inputs.ServiceTimeModel("lognormal", {"mu": math.log(13.0) - 0.5 ** 2 / 2, "sigma": 0.5})
    pool = np.random.default_rng(1).beta(2, 7, 5000)
    return experiments.Inputs(service_model=service, risk_pool=pool, session_minutes=T, slot_minutes=L)


@pytest.fixture(scope="module")
def s2(sim_inputs):
    return experiments.s2_policy_tradeoffs(sim_inputs, N_REPS, base_seed=3)


def row(table, **match):
    sel = table
    for key, value in match.items():
        sel = sel[sel[key] == value]
    assert len(sel) == 1, (match, len(sel))
    return sel.iloc[0]


# ---- S2 ---------------------------------------------------------------------------------------------------
def test_s2_has_one_row_per_policy_with_the_session_parameters(s2):
    assert list(s2["policy"]) == ["P0", "P1 k=2", "P1 k=3", "P1 k=4", "P1 k=6",
                                  "P2 0.3", "P2 0.5", "P2 p50", "P2 p70", "P2 p80", "P2 p90", "P2 p95"]
    assert (s2["session_minutes"] == T).all() and (s2["slot_minutes"] == L).all() and (s2["n_slots"] == N).all()
    assert (s2["n_reps"] == N_REPS).all() and (s2["base_seed"] == 3).all()


def test_s2_reports_every_metric_with_an_interval_and_paired_differences_against_p0(s2):
    for label in ("patients_served", "mean_wait_min", "overtime_min", "idle_min", "share_sessions_both_attend"):
        for suffix in ("", "_lo", "_hi", "_vs_p0", "_vs_p0_lo", "_vs_p0_hi"):
            assert f"{label}{suffix}" in s2.columns
        assert ((s2[f"{label}_lo"] <= s2[label] + 1e-12) & (s2[label] <= s2[f"{label}_hi"] + 1e-12)).all()
    p0 = row(s2, policy="P0")
    assert p0["patients_served_vs_p0"] == 0 and p0["patients_served_vs_p0_lo"] == 0 and p0["patients_served_vs_p0_hi"] == 0
    assert p0["double_slots_mean"] == 0 and p0["share_sessions_both_attend"] == 0


def test_s2_overbooking_serves_more_patients_at_the_price_of_waiting_and_overtime(s2):
    k2 = row(s2, policy="P1 k=2")
    assert k2["patients_served_vs_p0_lo"] > 0  # clearly more served, in every replication
    assert k2["mean_wait_min_vs_p0_lo"] > 0 and k2["overtime_min_vs_p0_lo"] > 0
    assert k2["idle_min_vs_p0_hi"] < 0  # and the doctor is idle less
    assert row(s2, policy="P1 k=6")["patients_served_vs_p0"] < k2["patients_served_vs_p0"]


def test_s2_higher_percentile_thresholds_double_book_fewer_slots(s2):
    p2 = s2[s2["family"] == "P2"].set_index("parameter")
    counts = [p2.loc[l, "double_slots_mean"] for l in ("p50", "p70", "p80", "p90", "p95")]
    assert counts == sorted(counts, reverse=True)
    assert p2.loc["p50", "double_slots_mean"] == pytest.approx(N * 0.5, rel=0.12)
    assert p2.loc["p95", "double_slots_mean"] == pytest.approx(N * 0.05, rel=0.45)


# ---- S3 ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s3(sim_inputs):
    return experiments.s3_probability_error(sim_inputs, N_REPS, base_seed=3)


def test_s3_without_a_shift_reproduces_s2_exactly(s2, s3):
    zero = s3[(s3["risk_shift"] == 0.0) & (s3["family"] == "P2")].set_index("policy")
    for policy in zero.index:
        for column in ("patients_served", "mean_wait_min", "overtime_min", "share_sessions_both_attend", "double_slots_mean"):
            assert zero.loc[policy, column] == row(s2, policy=policy)[column]
        assert zero.loc[policy, "patients_served_vs_unshifted"] == 0


def test_s3_a_biased_estimate_moves_who_gets_double_booked_but_not_the_thresholds(s3):
    assert sorted(set(s3["risk_shift"])) == [-0.10, -0.05, 0.0, 0.05, 0.10]
    assert len(s3[s3["family"] == "P2"]) == 5 * 7
    fixed = s3[(s3["policy"] == "P2 0.3")].set_index("risk_shift")
    assert fixed["threshold"].nunique() == 1  # the threshold is a design value, not recomputed on the shifted risks
    assert fixed.loc[0.10, "double_slots_mean"] > fixed.loc[0.0, "double_slots_mean"] > fixed.loc[-0.10, "double_slots_mean"]
    assert fixed.loc[0.10, "patients_served_vs_unshifted"] > 0 > fixed.loc[-0.10, "patients_served_vs_unshifted"]


# ---- S4 ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s4(sim_inputs):
    return experiments.s4_base_rate(sim_inputs, N_REPS, base_seed=3)


def test_s4_at_scale_one_reproduces_s2_and_a_higher_base_rate_raises_the_no_show_rate(s2, s4):
    one = s4[s4["base_rate_scale"] == 1.0].set_index("policy")
    for policy in s2["policy"]:
        assert one.loc[policy, "patients_served"] == row(s2, policy=policy)["patients_served"]
    rates = s4.groupby("base_rate_scale")["base_no_show_rate"].first()
    assert rates[1.5] == pytest.approx(3 * rates[0.5], rel=0.05) and rates[0.5] < rates[1.0] < rates[1.5]
    p0 = s4[s4["policy"] == "P0"].set_index("base_rate_scale")
    assert p0.loc[0.5, "patients_served"] > p0.loc[1.0, "patients_served"] > p0.loc[1.5, "patients_served"]


def test_s4_fixed_thresholds_stop_triggering_at_low_base_rates_but_percentiles_still_do(s4):
    low = s4[s4["base_rate_scale"] == 0.5].set_index("policy")
    assert low.loc["P2 0.5", "double_slots_mean"] < 0.1  # nobody is that risky any more
    assert low.loc["P2 p50", "double_slots_mean"] > N * 0.4  # the percentile rule adapts, as the plan intends


# ---- S5 ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s5(sim_inputs):
    return experiments.s5_service_variability(sim_inputs, N_REPS, base_seed=3)


def test_s5_changes_the_spread_not_the_mean_and_more_variability_means_longer_waits(s2, s5):
    one = s5[s5["variability_scale"] == 1.0].set_index("policy")
    assert one.loc["P1 k=2", "overtime_min"] == row(s2, policy="P1 k=2")["overtime_min"]
    spread = s5.groupby("variability_scale")[["service_mean_min", "service_sd_min"]].first()
    assert spread["service_mean_min"].max() == pytest.approx(spread["service_mean_min"].min(), rel=0.02)  # same mean
    assert spread.loc[1.5, "service_sd_min"] == pytest.approx(3 * spread.loc[0.5, "service_sd_min"], rel=0.05)
    p0 = s5[s5["policy"] == "P0"].set_index("variability_scale")
    assert p0.loc[1.5, "mean_wait_min"] > p0.loc[0.5, "mean_wait_min"]


# ---- S6 ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s6(sim_inputs):
    return experiments.s6_value_of_prediction(sim_inputs, 400, base_seed=3)


def test_s6_compares_selectors_in_two_matched_designs(s6):
    a, b = s6[s6["scheme"] == "matched_to_p2"], s6[s6["scheme"] == "matched_to_p1"]
    assert set(a["selector"]) == {"p2_risk_threshold", "random", "uniform", "oracle"} and a["setting"].nunique() == 7
    assert set(b["selector"]) == {"p1_uniform_k", "risk_top_m", "random", "oracle"} and b["setting"].nunique() == 4
    assert len(a) == 7 * 4 and len(b) == 4 * 4


def test_s6_every_selector_double_books_the_same_number_of_slots_and_serves_the_same_patients(s6):
    for (scheme, setting), group in s6.groupby(["scheme", "setting"]):
        assert group["double_slots_mean"].nunique() == 1, (scheme, setting)  # equal counts, session by session
        assert group["patients_served"].nunique() == 1, (scheme, setting)   # same extras, so the same number served
        assert (group["patients_served_vs_random"] == 0).all()
    k2 = s6[(s6["scheme"] == "matched_to_p1") & (s6["setting"] == "k=2")]
    assert (k2["double_slots_mean"] == N // 2).all()


def test_s6_the_oracle_is_a_ceiling_and_risk_ranking_beats_random_on_collisions(s6):
    for scheme, setting in {("matched_to_p2", s) for s in ("p70", "p80", "p90")} | {("matched_to_p1", "k=3"), ("matched_to_p1", "k=4")}:
        g = s6[(s6["scheme"] == scheme) & (s6["setting"] == setting)].set_index("selector")
        random, oracle = g.loc["random"], g.loc["oracle"]
        risk = g.loc["p2_risk_threshold" if scheme == "matched_to_p2" else "risk_top_m"]
        assert oracle["share_sessions_both_attend"] <= risk["share_sessions_both_attend"] + 1e-9
        assert oracle["share_sessions_both_attend"] < random["share_sessions_both_attend"]
        assert risk["share_double_slots_both_attend"] < random["share_double_slots_both_attend"]  # prediction does help here
        assert oracle["mean_wait_min_vs_random_hi"] < 0  # the oracle clearly waits less than random placement


def test_s6_the_comparison_against_random_is_paired_so_random_against_itself_is_exactly_zero(s6):
    random = s6[s6["selector"] == "random"]
    for label in ("mean_wait_min", "overtime_min", "idle_min", "share_sessions_both_attend"):
        assert (random[f"{label}_vs_random"] == 0).all() and (random[f"{label}_vs_random_hi"] == 0).all()


# ---- the whole run ----------------------------------------------------------------------------------------
def test_run_writes_every_table_and_figure_and_records_the_session_parameters_in_the_manifest(sim_inputs, tmp_path):
    (tmp_path / "raw.csv").write_text("stand-in for a raw data file\n")
    fit = inputs.fit_service_times(np.random.default_rng(5).lognormal(2.5, 0.5, 2000))
    results = tmp_path / "results"
    experiments.run(results_dir=results, n_reps=30, base_seed=3, repo_root=tmp_path,
                    prepared_inputs=(sim_inputs, fit, [tmp_path / "raw.csv"]))
    expected = ["sim_service_time_fit.csv", "s2_policy_tradeoffs.csv", "s3_probability_error.csv", "s4_base_rate.csv",
                "s5_service_variability.csv", "s6_value_of_prediction.csv"]
    for name in expected:
        assert (results / name).stat().st_size > 0, name
    for name in ("sim_service_time_qq.png", "sim_service_time_fit.png", "s2_tradeoffs.png", "s6_value_of_prediction.png"):
        assert (results / "figures" / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    stored = json.loads((results / "manifest.json").read_text())
    entry = stored["results/s2_policy_tradeoffs.csv"]
    assert entry["seed"] == 3 and entry["data_files"]
    note = json.loads(entry["note"])
    assert (note["session_minutes_T"], note["slot_minutes_L"], note["n_slots_N"], note["n_reps"]) == (T, L, N, 30)
    fit_table = pd.read_csv(results / "sim_service_time_fit.csv")
    assert fit_table["chosen"].sum() == 1 and "AIC" in fit_table.loc[fit_table["chosen"], "choice_reason"].iloc[0].upper()


# ---- the real inputs (need the raw data and the saved model) -------------------------------------------------------
needs_real = pytest.mark.skipif(not (data.HANGU_PATH.exists() and data.KAGGLE_PATH.exists()),
                                reason="raw Hangu and Kaggle files not in data/raw (see data/README.md)")


@needs_real
def test_the_real_defaults_are_computed_from_the_data():
    real, fit, files = experiments.build_inputs()
    assert real.slot_minutes % 5 == 0 and real.slot_minutes == inputs.default_slot_minutes(float(np.median(fit.minutes)))
    assert real.session_minutes == inputs.session_minutes_from_seed()
    assert real.session_minutes % real.slot_minutes == 0  # N = T / L is a whole number
    assert real.service_model.family == fit.chosen.family
    assert real.service_model.mean == pytest.approx(float(np.mean(fit.minutes)), rel=0.05)
    assert 20000 < len(real.risk_pool) < 25000 and ((real.risk_pool > 0) & (real.risk_pool < 1)).all()
    assert 0.15 < real.risk_pool.mean() < 0.25  # the deployable model's average predicted risk, near the base rate
    assert any(str(f).endswith("risk_model.joblib") for f in files)
