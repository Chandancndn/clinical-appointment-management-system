"""S1: does the simulator behave sensibly? One provider, one half-day session of T minutes, slots of L minutes.

Patients arrive exactly at their slot time, are seen first come first served, a no-show uses no doctor time.
(a) everyone attends and service = L: no waiting, no overtime   (b) nobody attends: nothing served, idle = T
(c) hand-worked cases with known waits                          (d) the same seed gives identical results
(e) overbooking never reduces the number served for the same random draws
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim import clinic, experiments, inputs, policies

T, L = 240, 15
N = T // L


def session(arrivals, attends, services, total=30):
    return clinic.simulate_session(arrivals, attends, services, total)


# ---- (a) a perfect day ---------------------------------------------------------------------------------
def test_everyone_attends_and_service_equals_the_slot_length_means_no_wait_and_no_overtime():
    arrivals = [k * L for k in range(N)]
    got = session(arrivals, [True] * N, [float(L)] * N, T)
    assert got.served == N
    assert got.mean_wait == 0.0 and max(got.waits) == 0.0
    assert got.overtime == 0.0
    assert got.idle == 0.0  # the doctor is busy for the whole session


# ---- (b) an empty day ----------------------------------------------------------------------------------
def test_nobody_attending_means_nothing_served_and_the_doctor_idle_for_the_whole_session():
    arrivals = [k * L for k in range(N)]
    got = session(arrivals, [False] * N, [float(L)] * N, T)
    assert got.served == 0
    assert got.idle == T
    assert got.overtime == 0.0
    assert math.isnan(got.mean_wait)  # nobody to wait


# ---- (c) hand-worked cases (slots at 0, 10, 20 in a 30-minute session) ---------------------------------
def test_hand_worked_three_patient_session_with_waits():
    # consultations 14, 8, 6 min.  P1 0-14.  P2 arrives 10, waits until 14, 14-22 (wait 4).  P3 arrives 20, 22-28 (wait 2).
    got = session([0, 10, 20], [True] * 3, [14, 8, 6])
    assert got.starts == (0, 14, 22) and got.ends == (14, 22, 28)
    assert got.waits == (0, 4, 2) and got.mean_wait == pytest.approx(2.0)
    assert got.overtime == 0.0
    assert got.idle == 2.0  # 28 busy minutes of 30
    assert got.served == 3


def test_hand_worked_session_that_runs_over():
    # consultations 14, 12, 9.  P2 14-26 (wait 4).  P3 arrives 20, 26-35 (wait 6): finishes 5 minutes after T = 30.
    got = session([0, 10, 20], [True] * 3, [14, 12, 9])
    assert got.waits == (0, 4, 6) and got.mean_wait == pytest.approx(10 / 3)
    assert got.ends == (14, 26, 35)
    assert got.overtime == 5.0
    assert got.idle == 0.0  # busy from 0 to 35, so the whole of 0-30


def test_hand_worked_session_with_a_no_show_uses_no_doctor_time():
    # P2 does not attend. P1 0-14, then the doctor waits; P3 arrives at 20, is seen at once, 20-29.
    got = session([0, 10, 20], [True, False, True], [14, 99, 9])  # the no-show's service time is never used
    assert got.served == 2 and got.waits == (0, 0)
    assert got.starts == (0, 20) and got.ends == (14, 29)
    assert got.idle == 7.0  # idle 14-20 and 29-30
    assert got.overtime == 0.0


def test_hand_worked_double_booked_slot_the_second_patient_waits_behind_the_first():
    # slot 0 holds two patients (both arrive at 0), slot 10 one more; consultations 14, 8, 6
    got = session([0, 0, 10], [True] * 3, [14, 8, 6])
    assert got.starts == (0, 14, 22) and got.waits == (0, 14, 12)
    assert got.mean_wait == pytest.approx(26 / 3)
    assert got.idle == 2.0 and got.overtime == 0.0


def test_hand_worked_double_booking_pays_off_when_the_first_patient_does_not_come():
    got = session([0, 0, 10], [False, True, True], [14, 8, 6])  # the extra patient takes the empty slot
    assert got.starts == (0, 10) and got.ends == (8, 16) and got.waits == (0, 0)
    assert got.served == 2 and got.idle == 16.0


def test_simpy_agrees_with_the_closed_form_recurrence_on_random_sessions():
    """start_i = max(arrival_i, end_{i-1}); an independent check of the SimPy model."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 25))
        arrivals = np.sort(rng.integers(0, 16, n) * 15.0)
        attends = rng.random(n) < 0.8
        services = rng.lognormal(2.4, 0.5, n)
        a = clinic.simulate_session(arrivals, attends, services, 240)
        b = clinic.simulate_session_recurrence(arrivals, attends, services, 240)
        assert a.served == b.served
        for x, y in ((a.mean_wait, b.mean_wait), (a.overtime, b.overtime), (a.idle, b.idle)):
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, abs=1e-9)
        assert np.allclose(a.waits, b.waits)


def test_arrivals_must_be_in_order():
    with pytest.raises(ValueError):
        session([10, 0], [True, True], [5, 5])


# ---- the simulation inputs -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_inputs():
    service = inputs.ServiceTimeModel("lognormal", {"mu": math.log(12.0) - 0.5 ** 2 / 2, "sigma": 0.5})
    pool = np.random.default_rng(1).beta(2, 7, 5000)
    return experiments.Inputs(service_model=service, risk_pool=pool, session_minutes=T, slot_minutes=L)


def arms(pool):
    return [policies.NoOverbooking(), *policies.p1_policies(), *policies.p2_policies(pool)]


# ---- (d) the same seed gives the same results ------------------------------------------------------------
def test_the_same_seed_gives_identical_results_and_a_different_seed_does_not(sim_inputs):
    a = experiments.run_arms(arms(sim_inputs.risk_pool), sim_inputs, n_reps=40, base_seed=7)
    b = experiments.run_arms(arms(sim_inputs.risk_pool), sim_inputs, n_reps=40, base_seed=7)
    c = experiments.run_arms(arms(sim_inputs.risk_pool), sim_inputs, n_reps=40, base_seed=8)
    for name in a:
        assert a[name].equals(b[name]), name
    assert not a["P0"].equals(c["P0"])


def test_replication_r_uses_seed_base_plus_r(sim_inputs):
    full = experiments.run_arms([policies.NoOverbooking()], sim_inputs, n_reps=10, base_seed=100)["P0"]
    for r in (0, 3, 9):
        alone = experiments.run_arms([policies.NoOverbooking()], sim_inputs, n_reps=1, base_seed=100 + r)["P0"]
        assert alone.iloc[0].equals(full.iloc[r])


def test_p0_does_not_change_when_other_policies_are_added(sim_inputs):
    only = experiments.run_arms([policies.NoOverbooking()], sim_inputs, n_reps=40, base_seed=5)["P0"]
    everything = experiments.run_arms(arms(sim_inputs.risk_pool), sim_inputs, n_reps=40, base_seed=5)["P0"]
    assert only.equals(everything)


# ---- (e) overbooking never reduces the number served ------------------------------------------------------
def test_overbooking_never_reduces_the_number_served_for_the_same_draws(sim_inputs):
    results = experiments.run_arms(arms(sim_inputs.risk_pool), sim_inputs, n_reps=150, base_seed=11)
    base = results["P0"]["served"].to_numpy()
    for name, frame in results.items():
        assert (frame["served"].to_numpy() >= base).all(), name
    assert results["P1 k=2"]["served"].mean() > base.mean() + 1  # and it really does add patients


def test_a_policy_that_overbooks_a_superset_of_slots_serves_at_least_as_many(sim_inputs):
    # every 4th slot is a subset of every 2nd slot, so k=2 never serves fewer than k=4 on the same draws
    r = experiments.run_arms([policies.Uniform(2), policies.Uniform(4), policies.Uniform(6)], sim_inputs, n_reps=150, base_seed=12)
    assert (r["P1 k=2"]["served"] >= r["P1 k=4"]["served"]).all()
    assert (r["P1 k=2"]["double_slots"] == N // 2).all() and (r["P1 k=4"]["double_slots"] == N // 4).all()


def test_overbooking_trades_waiting_and_overtime_for_patients_served(sim_inputs):
    r = experiments.run_arms([policies.NoOverbooking(), policies.Uniform(2)], sim_inputs, n_reps=150, base_seed=13)
    assert r["P1 k=2"]["mean_wait"].mean() > r["P0"]["mean_wait"].mean()
    assert r["P1 k=2"]["overtime"].mean() > r["P0"]["overtime"].mean()
    assert r["P1 k=2"]["idle"].mean() < r["P0"]["idle"].mean()


# ---- common random numbers ----------------------------------------------------------------------------------
def test_every_policy_sees_the_same_draws(sim_inputs):
    draw = experiments.make_draw(3, sim_inputs, base_seed=42)
    again = experiments.make_draw(3, sim_inputs, base_seed=42)
    for field in ("risk", "extra_risk", "u_primary", "u_extra", "service_primary", "service_extra", "random_scores"):
        assert np.array_equal(getattr(draw, field), getattr(again, field))
    # the same replication under another scale reuses the same uniforms: only the transform changes
    scaled = experiments.make_draw(3, sim_inputs, base_seed=42, base_rate_scale=0.5)
    assert np.array_equal(draw.u_primary, scaled.u_primary) and np.allclose(scaled.risk, draw.risk * 0.5)


def test_extra_patients_are_taken_in_order_so_policies_with_the_same_count_get_the_same_extras(sim_inputs):
    draw = experiments.make_draw(5, sim_inputs, base_seed=1)
    a = np.zeros(N, bool); a[[1, 5, 9]] = True
    b = np.zeros(N, bool); b[[2, 3, 14]] = True
    sa, sb = policies.build_session(draw, a, L), policies.build_session(draw, b, L)
    assert sum(sa.attends) - draw.primary_attends.sum() == sum(sb.attends) - draw.primary_attends.sum()
    assert np.isclose(sum(sa.services) - draw.service_primary.sum(), sum(sb.services) - draw.service_primary.sum())
    assert sa.double_slots == sb.double_slots == 3


# ---- policies -----------------------------------------------------------------------------------------------
def test_p1_double_books_every_kth_slot():
    draw = type("D", (), {})()
    for k, expected in ((2, [1, 3, 5, 7, 9, 11, 13, 15]), (3, [2, 5, 8, 11, 14]), (4, [3, 7, 11, 15]), (6, [5, 11])):
        mask = policies.Uniform(k).select(draw, np.zeros(N))
        assert list(np.flatnonzero(mask)) == expected


def test_p2_double_books_when_the_first_patients_predicted_risk_reaches_the_threshold():
    risk = np.array([0.1, 0.3, 0.29, 0.5, 0.9] + [0.0] * (N - 5))
    mask = policies.RiskThreshold(0.3, "0.3").select(None, risk)
    assert list(np.flatnonzero(mask)) == [1, 3, 4]  # >= 0.3, inclusive


def test_p2_grid_has_two_fixed_thresholds_and_five_percentiles_of_the_risk_distribution():
    pool = np.random.default_rng(2).beta(2, 7, 20000)
    grid = policies.p2_policies(pool)
    assert [p.label for p in grid] == ["0.3", "0.5", "p50", "p70", "p80", "p90", "p95"]
    thresholds = {p.label: p.threshold for p in grid}
    assert thresholds["0.3"] == 0.3 and thresholds["0.5"] == 0.5
    for q in (50, 70, 80, 90, 95):
        assert thresholds[f"p{q}"] == pytest.approx(np.percentile(pool, q))
        assert (pool >= thresholds[f"p{q}"]).mean() == pytest.approx(1 - q / 100, abs=0.01)  # triggers for the top (100-q)%


def test_ranked_selectors_pick_exactly_the_matched_number_of_slots(sim_inputs):
    draw = experiments.make_draw(2, sim_inputs, base_seed=9)
    for selector in ("risk", "random", "uniform", "oracle"):
        for m in (0, 1, 4, N):
            mask = policies.Ranked(selector).select(draw, draw.risk, matched_count=m)
            assert mask.sum() == m, (selector, m)


def test_the_oracle_double_books_the_slots_whose_first_patient_will_not_come():
    draw = type("D", (), {})()
    draw.primary_attends = np.array([True, False, True, False, True, True, False, True] + [True] * (N - 8))
    draw.random_scores = np.zeros(N)
    risk = np.linspace(0, 0.5, N)
    mask = policies.Ranked("oracle").select(draw, risk, matched_count=2)
    assert set(np.flatnonzero(mask)) <= {1, 3, 6}  # both chosen slots are real no-shows
    # with fewer no-shows than slots to fill, it takes every no-show first
    mask = policies.Ranked("oracle").select(draw, risk, matched_count=5)
    assert {1, 3, 6} <= set(np.flatnonzero(mask))


def test_the_uniform_selector_spreads_slots_evenly_through_the_session():
    mask = policies.Ranked("uniform").select(None, np.zeros(N), matched_count=4)
    chosen = np.flatnonzero(mask)
    assert len(chosen) == 4 and np.all(np.diff(chosen) == N // 4)


# ---- service times: fitting, choice, variability --------------------------------------------------------------
def test_default_slot_length_is_the_median_rounded_up_to_the_next_five_minutes():
    assert inputs.default_slot_minutes(12.08) == 15
    assert inputs.default_slot_minutes(10.0) == 10
    assert inputs.default_slot_minutes(10.01) == 15
    assert inputs.default_slot_minutes(4.2) == 5


def test_the_default_session_length_comes_from_the_seeded_app_slots():
    from datetime import date, datetime

    from db import seed_synthetic

    start, end = seed_synthetic.SESSION
    expected = (datetime.combine(date.min, end) - datetime.combine(date.min, start)).seconds // 60
    assert inputs.session_minutes_from_seed() == expected == 240


def test_the_fit_chooses_the_family_with_the_lower_aic():
    rng = np.random.default_rng(3)
    logn = rng.lognormal(2.4, 0.9, 6000)  # strongly skewed: lognormal's heavier tail wins
    gamm = rng.gamma(9.0, 1.5, 6000)      # near-symmetric: gamma wins
    assert inputs.fit_service_times(logn).chosen.family == "lognormal"
    assert inputs.fit_service_times(gamm).chosen.family == "gamma"


def test_the_fit_reports_both_families_with_parameters_and_aic():
    fit = inputs.fit_service_times(np.random.default_rng(4).lognormal(2.4, 0.5, 3000))
    table = fit.table()
    assert list(table["family"]) == ["lognormal", "gamma"]
    assert {"param_1_name", "param_1", "param_2_name", "param_2", "log_likelihood", "aic", "chosen", "n"} <= set(table.columns)
    assert table["aic"].idxmin() == table["chosen"].idxmax()
    assert table["n"].iloc[0] == 3000


@pytest.mark.parametrize("family, params", [("lognormal", {"mu": 2.4, "sigma": 0.5}), ("gamma", {"shape": 5.0, "scale": 2.5})])
def test_scaling_variability_keeps_the_mean_and_scales_the_standard_deviation(family, params):
    model = inputs.ServiceTimeModel(family, params)
    grid = np.linspace(0.0005, 0.9995, 4000)  # quantiles of the distribution
    base = model.ppf(grid, variability=1.0)
    for scale in (0.5, 1.5):
        scaled = model.ppf(grid, variability=scale)
        assert scaled.mean() == pytest.approx(base.mean(), rel=0.03)
        assert scaled.std() == pytest.approx(scale * base.std(), rel=0.05)
    assert model.mean == pytest.approx(base.mean(), rel=0.03)
    assert np.array_equal(model.ppf(grid, 1.0), base)  # variability 1 changes nothing


# ---- confidence intervals ---------------------------------------------------------------------------------------
def test_mean_ci_is_the_t_interval_and_ignores_missing_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    mean, lo, hi = experiments.mean_ci(x)
    assert mean == 2.5 and lo < 2.5 < hi
    m2, lo2, hi2 = experiments.mean_ci(np.array([5.0, 5.0, 5.0]))
    assert (m2, lo2, hi2) == (5.0, 5.0, 5.0)


def test_paired_differences_are_zero_against_oneself_and_ignore_pairs_with_a_missing_value():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert experiments.paired_ci(a, a) == (0.0, 0.0, 0.0)
    mean, lo, hi = experiments.paired_ci(b + 1, b)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)
    assert experiments.paired_ci(a, b)[0] == 0.0  # the NaN pair is dropped, the rest are equal
