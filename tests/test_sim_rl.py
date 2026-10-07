"""S7: a booking policy learned by reinforcement learning (the plan's stretch item), on small synthetic inputs.

The learned policy (P3) decides slot by slot whether to add a second patient. It sees exactly what the risk-threshold
policy P2 sees (the predicted risk of the slot's first patient) plus its own earlier choices, is trained by policy
gradient (REINFORCE) on simulated sessions for a STATED cost of waiting and overtime, and is then judged on replications it
never trained on, with the full trade-off metrics, not only its own reward.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim import experiments, inputs, policies, rl

T, L = 240, 15
N = T // L


@pytest.fixture(scope="module")
def sim_inputs():
    service = inputs.ServiceTimeModel("lognormal", {"mu": math.log(13.0) - 0.5 ** 2 / 2, "sigma": 0.5})
    pool = np.random.default_rng(1).beta(2, 7, 5000)
    return experiments.Inputs(service_model=service, risk_pool=pool, session_minutes=T, slot_minutes=L)


@pytest.fixture(scope="module")
def scaling(sim_inputs):
    return rl.Scaling.from_pool(sim_inputs.risk_pool)


def policy(theta, scaling, label="test"):
    return rl.LearnedPolicy(np.array(theta, dtype=float), scaling, label)


# ---- the reward is a stated weighting, nothing hidden -----------------------------------------------------------
def test_the_reward_is_patients_served_minus_the_stated_costs_of_waiting_and_overtime():
    costs = rl.Costs("example", wait_per_hour=1.0, overtime_per_hour=2.0)
    # 10 served, 6 minutes mean wait (one patient-hour in total), 30 minutes overtime: 10 - 1 - 1
    assert rl.reward(served=10, mean_wait=6.0, overtime=30.0, costs=costs) == pytest.approx(8.0)
    assert rl.reward(served=0, mean_wait=math.nan, overtime=0.0, costs=costs) == 0.0  # an empty session waits for nobody
    assert rl.reward(served=10, mean_wait=6.0, overtime=30.0, costs=rl.Costs("free", 0.0, 0.0)) == 10.0


def test_three_named_cost_settings_are_fixed_in_the_code():
    assert [c.name for c in rl.COST_SETTINGS] == ["light", "moderate", "heavy"]
    assert all(c.wait_per_hour > 0 and c.overtime_per_hour > 0 for c in rl.COST_SETTINGS)
    waits = [c.wait_per_hour for c in rl.COST_SETTINGS]
    assert waits == sorted(waits) and len(set(waits)) == 3


# ---- what the policy sees ---------------------------------------------------------------------------------------
def test_the_features_are_the_bias_the_risk_the_position_and_the_policys_own_earlier_choices(scaling):
    first = rl.step_features(risk=scaling.risk_mean, slot=0, n_slots=N, doubled=0, load=0.0, scaling=scaling)
    last = rl.step_features(risk=scaling.risk_mean, slot=N - 1, n_slots=N, doubled=0, load=0.0, scaling=scaling)
    assert len(first) == len(rl.FEATURES) == 5 and first[0] == 1.0
    assert first[1] == pytest.approx(0.0)  # risk is centred on the pool mean
    assert first[2] == pytest.approx(-1.0) and last[2] == pytest.approx(1.0)
    more = rl.step_features(risk=scaling.risk_mean + scaling.risk_sd, slot=5, n_slots=N, doubled=3, load=1.5, scaling=scaling)
    base = rl.step_features(risk=scaling.risk_mean, slot=5, n_slots=N, doubled=0, load=0.0, scaling=scaling)
    assert more[1] == pytest.approx(1.0) and more[3] > base[3] and more[4] > base[4]


def test_the_policy_decides_from_the_visible_risk_alone_and_returns_one_decision_per_slot(sim_inputs, scaling):
    draw = experiments.make_draw(0, sim_inputs, 3)
    learned = policy([0.2, 1.0, -0.3, -0.5, -0.4], scaling)
    mask = learned.select(draw, draw.risk)
    assert mask.dtype == bool and mask.shape == (N,)
    assert (learned.select(None, draw.risk) == mask).all()  # it never looks at attendance, service times or the extras
    assert (learned.select(draw, draw.risk) == mask).all()  # and it is deterministic


def test_extreme_weights_double_book_everything_or_nothing(sim_inputs, scaling):
    draw = experiments.make_draw(1, sim_inputs, 3)
    assert policy([50, 0, 0, 0, 0], scaling).select(draw, draw.risk).all()
    assert not policy([-50, 0, 0, 0, 0], scaling).select(draw, draw.risk).any()


def test_a_decision_never_depends_on_later_slots(sim_inputs, scaling):
    learned = policy([0.0, 1.5, 0.2, -1.0, -0.8], scaling)
    risk = experiments.make_draw(2, sim_inputs, 3).risk
    for cut in (3, 8, 12):
        changed = risk.copy()
        changed[cut:] = 0.99
        assert (learned.select(None, changed)[:cut] == learned.select(None, risk)[:cut]).all()


def test_every_risk_threshold_policy_is_a_special_case(sim_inputs, scaling):
    """With weight only on the risk, P3 is P2: so anything P2 can do, the learned policy could also do."""
    threshold = float(np.percentile(sim_inputs.risk_pool, 80))
    steep = 1e6
    z = (threshold - scaling.risk_mean) / scaling.risk_sd
    as_p3 = policy([-steep * z, steep, 0, 0, 0], scaling)
    p2 = policies.RiskThreshold(threshold, "p80")
    for r in range(50):
        draw = experiments.make_draw(r, sim_inputs, 3)
        assert (as_p3.select(draw, draw.risk) == p2.select(draw, draw.risk)).all()


# ---- the gradient the learner follows is the real gradient ------------------------------------------------------
def test_the_sampled_gradient_is_the_gradient_of_the_log_probability(sim_inputs, scaling):
    theta = np.array([-0.4, 0.8, 0.3, -0.6, 0.5])
    learned = policy(theta, scaling)
    risk = experiments.make_draw(4, sim_inputs, 3).risk
    mask, gradient = learned.sample(risk, np.random.default_rng(0))
    numeric = np.zeros(5)
    for k in range(5):
        step = np.zeros(5)
        step[k] = 1e-5
        numeric[k] = (policy(theta + step, scaling).log_prob(risk, mask) - policy(theta - step, scaling).log_prob(risk, mask)) / 2e-5
    assert gradient == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_sampling_follows_the_policys_probabilities(scaling):
    always = policy([50, 0, 0, 0, 0], scaling)
    mask, _ = always.sample(np.full(N, 0.2), np.random.default_rng(0))
    assert mask.all()
    coin = policy([0, 0, 0, 0, 0], scaling)  # every decision is a fair coin
    share = np.mean([coin.sample(np.full(N, 0.2), np.random.default_rng(seed))[0].mean() for seed in range(200)])
    assert 0.45 < share < 0.55


# ---- the fast session used in training is the SimPy session ------------------------------------------------------
def test_the_fast_session_gives_the_same_numbers_as_the_simpy_one(sim_inputs):
    for r in range(25):
        draw = experiments.make_draw(r, sim_inputs, 3)
        mask = policies.Uniform(2 + r % 3).select(draw, draw.risk)
        slow, fast = experiments.evaluate_mask(draw, mask, sim_inputs), rl.fast_metrics(draw, mask, sim_inputs)
        for key in ("served", "overtime", "idle", "double_slots", "both_slots"):
            assert fast[key] == pytest.approx(slow[key]), (r, key)
        assert (math.isnan(fast["mean_wait"]) and math.isnan(slow["mean_wait"])) or fast["mean_wait"] == pytest.approx(slow["mean_wait"])


# ---- training ----------------------------------------------------------------------------------------------------
SMALL = dict(iterations=60, batch=64, validate_every=10, n_validation=200)


def doubled_share(learned, sim_inputs, seeds=range(5000, 5100)) -> float:
    return float(np.mean([learned.select(None, experiments.make_draw(r, sim_inputs, 3).risk).mean() for r in seeds]))


def test_with_no_cost_of_delay_the_learner_double_books_almost_everything(sim_inputs):
    learned, log = rl.train(sim_inputs, rl.Costs("free", 0.0, 0.0), base_seed=3, **SMALL)
    assert doubled_share(learned, sim_inputs) > 0.9
    assert log["validation_reward"].iloc[-1] > log["validation_reward"].iloc[0]


def test_with_a_huge_cost_of_delay_the_learner_stops_double_booking(sim_inputs, scaling):
    everything = (3.0, 0.0, 0.0, 0.0, 0.0)  # start from a policy that double-books every slot, so it has to unlearn it
    assert doubled_share(policy(everything, scaling), sim_inputs) == 1.0
    learned, log = rl.train(sim_inputs, rl.Costs("punishing", 50.0, 100.0), base_seed=3, initial_weights=everything,
                            **{**SMALL, "iterations": 150})
    assert doubled_share(learned, sim_inputs) < 0.05
    assert log["validation_reward"].max() > log["validation_reward"].iloc[0]


def test_between_the_extremes_a_higher_cost_of_delay_means_fewer_double_booked_slots(sim_inputs):
    shares = [doubled_share(rl.train(sim_inputs, costs, base_seed=3, index=i, **{**SMALL, "iterations": 120})[0], sim_inputs)
              for i, costs in enumerate((rl.Costs("cheap", 0.1, 0.2), rl.Costs("dear", 6.0, 12.0)))]
    assert shares[0] > shares[1]


def test_training_is_repeatable_and_depends_on_the_seed(sim_inputs):
    costs = rl.Costs("moderate", 1.0, 2.0)
    quick = dict(iterations=12, batch=32, validate_every=4, n_validation=50)
    a, log_a = rl.train(sim_inputs, costs, base_seed=3, **quick)
    b, log_b = rl.train(sim_inputs, costs, base_seed=3, **quick)
    c, _ = rl.train(sim_inputs, costs, base_seed=4, **quick)
    assert (a.theta == b.theta).all() and log_a.equals(log_b)
    assert not (a.theta == c.theta).all()


def test_the_policy_kept_is_the_one_that_did_best_on_the_validation_sessions(sim_inputs):
    learned, log = rl.train(sim_inputs, rl.Costs("moderate", 1.0, 2.0), base_seed=3, iterations=30, batch=32,
                            validate_every=5, n_validation=100)
    best = log.loc[log["validation_reward"].idxmax()]
    assert bool(best["kept"]) and log["kept"].sum() == 1
    assert learned.theta == pytest.approx(best[[f"w_{name}" for name in rl.FEATURES]].to_numpy(dtype=float))


def test_training_validation_and_evaluation_never_share_a_session():
    plan = rl.seed_plan(base_seed=42, n_reps=1000, iterations=400, batch=256, n_validation=1000)
    evaluation, training, validation = plan["evaluation"], plan["training"], plan["validation"]
    assert evaluation == range(42, 1042)
    for a, b in ((evaluation, training), (evaluation, validation), (training, validation)):
        assert a.stop <= b.start or b.stop <= a.start
    assert len(training) == 400 * 256 and len(validation) == 1000


# ---- the experiment tables ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s7(sim_inputs):
    learned = {costs: rl.train(sim_inputs, costs, base_seed=3, iterations=20, batch=32, validate_every=5, n_validation=80,
                               index=i)[0] for i, costs in enumerate(rl.COST_SETTINGS)}
    return rl.evaluate(sim_inputs, learned, n_reps=120, base_seed=3)


def test_the_learned_policies_are_reported_with_every_metric_not_only_their_reward(s7):
    table = s7["s7_learned_policy"]
    assert list(table["policy"]) == ["P3 light", "P3 moderate", "P3 heavy"] and (table["family"] == "P3").all()
    for label in ("patients_served", "mean_wait_min", "overtime_min", "idle_min", "share_sessions_both_attend"):
        for suffix in ("", "_lo", "_hi", "_vs_p0", "_vs_p0_lo", "_vs_p0_hi"):
            assert f"{label}{suffix}" in table.columns
    for column in ("cost_setting", "wait_cost_per_hour", "overtime_cost_per_hour", "reward", "reward_lo", "reward_hi",
                   "best_standard_policy", "reward_vs_best_standard", "reward_vs_best_standard_lo", "reward_vs_best_standard_hi",
                   "double_slots_mean", "n_reps", "base_seed") + tuple(f"w_{name}" for name in rl.FEATURES):
        assert column in table.columns, column
    assert ((table["reward_lo"] <= table["reward"]) & (table["reward"] <= table["reward_hi"])).all()
    assert list(table["wait_cost_per_hour"]) == [c.wait_per_hour for c in rl.COST_SETTINGS]


def test_every_policy_is_scored_under_every_cost_setting(s7):
    table = s7["s7_reward_by_policy"]
    standard = ["P0", "P1 k=2", "P1 k=3", "P1 k=4", "P1 k=6", "P2 0.3", "P2 0.5", "P2 p50", "P2 p70", "P2 p80", "P2 p90", "P2 p95"]
    for costs in rl.COST_SETTINGS:
        part = table[table["cost_setting"] == costs.name]
        assert list(part["policy"]) == standard + [f"P3 {costs.name}"]
        learned = part[part["policy"] == f"P3 {costs.name}"].iloc[0]
        assert learned["reward_vs_learned"] == 0 and learned["reward_vs_learned_lo"] == 0
        assert part["is_best_standard"].sum() == 1
        best = part[part["is_best_standard"]].iloc[0]
        assert best["reward"] == part[part["family"] != "P3"]["reward"].max()


def test_at_the_same_number_of_double_booked_slots_every_way_of_choosing_serves_the_same_patients(s7):
    table = s7["s7_matched"]
    assert set(table["scheme"]) == {"matched_to_p3"}
    for costs in rl.COST_SETTINGS:
        part = table[table["setting"] == costs.name]
        assert list(part["selector"]) == ["p3_learned", "risk_top_m", "random", "uniform", "oracle"]
        assert part["double_slots_mean"].nunique() == 1 and part["patients_served"].nunique() == 1
        assert (part["patients_served_vs_uniform"].abs() < 1e-12).all()
        assert "mean_wait_min_vs_uniform_lo" in part.columns and "overtime_min_vs_random_hi" in part.columns


def test_the_learned_policies_are_also_tried_with_a_biased_risk_estimate(s7):
    table = s7["s7_probability_error"]
    assert sorted(set(table["risk_shift"])) == sorted(experiments.SHIFTS)
    assert len(table) == len(experiments.SHIFTS) * len(rl.COST_SETTINGS)
    assert {"policy", "cost_setting", "reward", "double_slots_mean", "mean_wait_min", "overtime_min"} <= set(table.columns)
    unshifted = table[table["risk_shift"] == 0.0].set_index("policy")["patients_served"]
    main = s7["s7_learned_policy"].set_index("policy")["patients_served"]
    assert unshifted.to_dict() == pytest.approx(main.to_dict())  # the same sessions, the same numbers


def test_the_standard_policies_are_the_same_sessions_as_s2(sim_inputs, s7):
    s2 = experiments.s2_policy_tradeoffs(sim_inputs, 120, base_seed=3).set_index("policy")
    light = s7["s7_reward_by_policy"]
    light = light[light["cost_setting"] == "light"].set_index("policy")
    for name in ("P0", "P1 k=3", "P2 p80"):
        assert light.loc[name, "patients_served"] == pytest.approx(s2.loc[name, "patients_served"])
        assert light.loc[name, "mean_wait_min"] == pytest.approx(s2.loc[name, "mean_wait_min"])


# ---- the whole run ---------------------------------------------------------------------------------------------------
def test_run_writes_the_s7_files_and_records_how_the_policies_were_trained(sim_inputs, tmp_path):
    import json

    import pandas as pd

    (tmp_path / "raw.csv").write_text("stand-in for a raw data file\n")
    results = tmp_path / "results"
    quick = {"iterations": 10, "batch": 16, "validate_every": 5, "n_validation": 40}
    rl.run(results_dir=results, n_reps=40, base_seed=3, repo_root=tmp_path, prepared_inputs=(sim_inputs, None, [tmp_path / "raw.csv"]),
           training=quick)
    names = ["s7_learned_policy.csv", "s7_reward_by_policy.csv", "s7_matched.csv", "s7_probability_error.csv", "s7_training.csv"]
    for name in names:
        assert (results / name).stat().st_size > 0, name
    assert (results / "figures" / "s7_learned_policy.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    stored = json.loads((results / "manifest.json").read_text())
    for name in names + ["figures/s7_learned_policy.png"]:
        assert stored[f"results/{name}"]["script"] == "sim/rl.py" and stored[f"results/{name}"]["seed"] == 3, name
    note = json.loads(stored["results/s7_learned_policy.csv"]["note"])
    assert note["training"]["iterations"] == 10 and note["cost_settings"]["moderate"] == {"wait_per_hour": 1.0, "overtime_per_hour": 2.0}
    assert note["features"] == list(rl.FEATURES) and note["n_reps"] == 40
    log = pd.read_csv(results / "s7_training.csv")
    assert set(log["cost_setting"]) == {"light", "moderate", "heavy"} and log.groupby("cost_setting")["kept"].sum().eq(1).all()
    assert not any((results / name).exists() for name in ("s2_policy_tradeoffs.csv", "s6_value_of_prediction.csv"))  # S2 to S6 untouched
