"""S7: a booking policy learned by reinforcement learning.   python -m sim.rl

The plan's stretch item, in the spirit of Amalina and An (2026, an arXiv preprint): learn where to double-book instead
of fixing a rule by hand. This is a small policy-gradient learner, not a reproduction of that paper.

THE POLICY (P3). Slot by slot, in booking order, it decides whether the slot gets a second patient. It sees exactly
what the risk-threshold policy P2 sees, the predicted risk of the slot's first patient, plus its own earlier choices:

    bias       1
    risk       the first patient's predicted risk, centred and scaled by the risk pool
    position   where the slot is in the session, -1 (first) to +1 (last)
    doubled    how many earlier slots it has already double-booked
    load       patients it expects to have arrived so far, minus the slots that have passed

It double-books with probability sigmoid(weights . features). With weight on the risk only it IS a P2 threshold, so
anything P2 can do, P3 could also do; the other weights let it space its choices and react to how full the session is.
It never sees attendance, consultation times or the extra patients.

LEARNING. REINFORCE with a baseline: sample sessions, sample the decisions, simulate, and move the weights towards the
decisions that scored better than not overbooking at all on the same session (that is the baseline; it removes most of
the luck of the draw). Adam steps, a fixed seed, and the weights kept are the ones that did best on validation sessions.

THE REWARD IS A STATED CHOICE. Learning needs one number per session, so a cost has to be put on delay:

    reward = patients served - wait_cost x (patient-hours waited) - overtime_cost x (doctor overtime hours)

There is no right value for those costs (LaGanga and Lawrence, 2007), so three settings are fixed in COST_SETTINGS
and a policy is learned for each. The results still report every metric (patients served, waiting, overtime, idle time,
collisions) for each learned policy, exactly like S2, and compare it with the hand-made rules at the SAME number of
double-booked slots, exactly like S6. The reward is only how the policy was trained and one extra column.

HONEST EVALUATION. Training sessions, validation sessions and the 1,000 evaluation sessions use disjoint seeds
(seed_plan), and the evaluation sessions are the same ones S2 to S6 use, so every comparison is paired.

Writes results/s7_learned_policy.csv, s7_reward_by_policy.csv, s7_matched.csv, s7_probability_error.csv,
s7_training.csv and figures/s7_learned_policy.png, each recorded in results/manifest.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src import data
from ml.src.experiment import ResultWriter

from . import clinic, experiments, policies

SCRIPT = "sim/rl.py"
FEATURES = ("bias", "risk", "position", "doubled", "load")
TRAIN_OFFSET = 1_000_000  # training sessions use seeds base + 1,000,000 + k
VALIDATION_OFFSET = 50_000_000  # validation sessions use seeds base + 50,000,000 + j
INITIAL_WEIGHTS = (-1.0, 0.0, 0.0, 0.0, 0.0)  # starts by double-booking about a quarter of the slots, whatever the risk
TRAINING = {"iterations": 600, "batch": 256, "learning_rate": 0.05, "validate_every": 10, "n_validation": 1000}
SELECTORS = ["p3_learned", "risk_top_m", "random", "uniform", "oracle"]


@dataclass(frozen=True)
class Costs:
    """What delay costs, in patients served: per patient-hour of waiting and per hour of doctor overtime."""

    name: str
    wait_per_hour: float
    overtime_per_hour: float


# Three stated assumptions, from "delay matters little" to "delay matters a lot". None of them is the right one.
COST_SETTINGS = (Costs("light", 0.5, 1.0), Costs("moderate", 1.0, 2.0), Costs("heavy", 2.0, 4.0))


@dataclass(frozen=True)
class Scaling:
    """What the policy knows about the predicted-risk distribution: its mean and spread."""

    risk_mean: float
    risk_sd: float

    @classmethod
    def from_pool(cls, pool) -> "Scaling":
        pool = np.asarray(pool, dtype=float)
        return cls(float(pool.mean()), float(pool.std()) or 1.0)

    @property
    def extra_attend(self) -> float:
        """The chance an extra patient attends, as far as the policy knows: one minus the average risk."""
        return 1.0 - self.risk_mean


# ---- the reward -------------------------------------------------------------------------------------------------
def reward(served, mean_wait, overtime, costs: Costs) -> float:
    waited_hours = 0.0 if not served or (isinstance(mean_wait, float) and math.isnan(mean_wait)) else served * mean_wait / 60.0
    return float(served - costs.wait_per_hour * waited_hours - costs.overtime_per_hour * overtime / 60.0)


def reward_series(frame: pd.DataFrame, costs: Costs) -> np.ndarray:
    """The reward of every replication in a frame of session metrics."""
    waited_hours = (frame["served"] * frame["mean_wait"].fillna(0.0) / 60.0).to_numpy(dtype=float)
    return (frame["served"].to_numpy(dtype=float) - costs.wait_per_hour * waited_hours
            - costs.overtime_per_hour * frame["overtime"].to_numpy(dtype=float) / 60.0)


# ---- the policy -------------------------------------------------------------------------------------------------
def step_features(risk: float, slot: int, n_slots: int, doubled: int, load: float, scaling: Scaling) -> list:
    position = 2.0 * slot / (n_slots - 1) - 1.0 if n_slots > 1 else 0.0
    return [1.0, (risk - scaling.risk_mean) / scaling.risk_sd, position, 4.0 * doubled / n_slots, load / 2.0]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    e = math.exp(max(z, -700.0))
    return e / (1.0 + e)


def _log_sigmoid(z: float) -> float:
    return -(max(-z, 0.0) + math.log1p(math.exp(-abs(z))))  # log sigmoid(z), stable for large |z|


class LearnedPolicy:
    family, threshold = "P3", None

    def __init__(self, theta, scaling: Scaling, label: str) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.scaling, self.label, self.name = scaling, label, f"P3 {label}"

    def _walk(self, visible_risk, choose):
        """One pass through the session. `choose(slot, probability)` picks the action; returns the decisions, the gradient
        of their log-probability with respect to the weights, and that log-probability."""
        weights, n = self.theta.tolist(), len(visible_risk)
        mask = np.zeros(n, dtype=bool)
        gradient, log_probability = [0.0] * len(FEATURES), 0.0
        doubled, load = 0, 0.0
        for slot in range(n):
            risk = float(visible_risk[slot])
            features = step_features(risk, slot, n, doubled, load, self.scaling)
            z = sum(w * f for w, f in zip(weights, features))
            probability = _sigmoid(z)
            action = bool(choose(slot, probability))
            log_probability += _log_sigmoid(z) if action else _log_sigmoid(-z)
            residual = (1.0 if action else 0.0) - probability  # d log pi / dz for a logistic policy
            for k, feature in enumerate(features):
                gradient[k] += residual * feature
            if action:
                mask[slot] = True
                doubled += 1
            load += (1.0 - risk) + (self.scaling.extra_attend if action else 0.0) - 1.0
        return mask, np.array(gradient), log_probability

    def select(self, draw, visible_risk, matched_count=None) -> np.ndarray:
        """The decisions used for evaluation: double-book where the probability is at least one half. `draw` is ignored."""
        return self._walk(visible_risk, lambda slot, probability: probability >= 0.5)[0]

    def sample(self, visible_risk, rng):
        """Decisions drawn from the policy's probabilities (training), and the gradient of their log-probability."""
        uniforms = rng.random(len(visible_risk))
        mask, gradient, _ = self._walk(visible_risk, lambda slot, probability: uniforms[slot] < probability)
        return mask, gradient

    def log_prob(self, visible_risk, mask) -> float:
        return self._walk(visible_risk, lambda slot, probability: bool(mask[slot]))[2]


# ---- sessions for training --------------------------------------------------------------------------------------
def fast_metrics(draw: policies.Draw, mask: np.ndarray, inputs: experiments.Inputs) -> dict:
    """The same metrics as experiments.evaluate_mask, from the single-server recurrence instead of SimPy (a test checks
    they agree). Training simulates hundreds of thousands of sessions, so it uses this; evaluation uses SimPy."""
    session = policies.build_session(draw, mask, inputs.slot_minutes)
    result = clinic.simulate_session_recurrence(session.arrivals, session.attends, session.services, inputs.session_minutes)
    return {"served": result.served, "mean_wait": result.mean_wait, "overtime": result.overtime, "idle": result.idle,
            "both_attend": float(session.both_slots > 0), "double_slots": session.double_slots,
            "both_slots": session.both_slots}


def _session_reward(draw, mask, inputs, costs) -> float:
    metrics = fast_metrics(draw, mask, inputs)
    return reward(metrics["served"], metrics["mean_wait"], metrics["overtime"], costs)


def seed_plan(base_seed: int, n_reps: int, iterations: int, batch: int, n_validation: int) -> dict:
    """The seeds each stage uses. They never overlap, so the policy is judged on sessions it has not seen."""
    return {"evaluation": range(base_seed, base_seed + n_reps),
            "training": range(base_seed + TRAIN_OFFSET, base_seed + TRAIN_OFFSET + iterations * batch),
            "validation": range(base_seed + VALIDATION_OFFSET, base_seed + VALIDATION_OFFSET + n_validation)}


def train(inputs: experiments.Inputs, costs: Costs, base_seed: int = experiments.BASE_SEED, iterations: int = TRAINING["iterations"],
          batch: int = TRAINING["batch"], learning_rate: float = TRAINING["learning_rate"],
          validate_every: int = TRAINING["validate_every"], n_validation: int = TRAINING["n_validation"], index: int = 0,
          initial_weights=INITIAL_WEIGHTS):
    """Learn the weights for one cost setting. Returns (the policy that did best on the validation sessions, the log).

    `index` only separates the action-sampling random stream of one cost setting from another's. `initial_weights` is where
    learning starts (the experiments always use INITIAL_WEIGHTS; a test starts elsewhere to see the learner move).
    """
    if TRAIN_OFFSET + iterations * batch > VALIDATION_OFFSET:
        raise ValueError("too many training sessions: they would run into the validation seeds")
    scaling = Scaling.from_pool(inputs.risk_pool)
    rng = np.random.default_rng([base_seed, 977, index])
    theta = np.array(initial_weights, dtype=float)
    nothing = np.zeros(inputs.n_slots, dtype=bool)

    validation = [experiments.make_draw(VALIDATION_OFFSET + j, inputs, base_seed) for j in range(n_validation)]

    def validate(weights) -> tuple:
        candidate = LearnedPolicy(weights, scaling, costs.name)
        masks = [candidate.select(None, draw.risk) for draw in validation]
        rewards = [_session_reward(draw, mask, inputs, costs) for draw, mask in zip(validation, masks)]
        return float(np.mean(rewards)), float(np.mean([mask.sum() for mask in masks]))

    first_moment, second_moment = np.zeros_like(theta), np.zeros_like(theta)
    rows = []

    def log(iteration: int, train_reward: float, train_advantage: float) -> None:
        value, doubled = validate(theta)
        rows.append({"iteration": iteration, "sessions": iteration * batch, "train_reward": train_reward,
                     "train_reward_vs_p0": train_advantage, "validation_reward": value, "validation_double_slots": doubled,
                     **{f"w_{name}": float(w) for name, w in zip(FEATURES, theta)}})

    log(0, math.nan, math.nan)
    for iteration in range(1, iterations + 1):
        policy = LearnedPolicy(theta, scaling, costs.name)
        gradients, advantages, rewards = [], [], []
        for b in range(batch):
            draw = experiments.make_draw(TRAIN_OFFSET + (iteration - 1) * batch + b, inputs, base_seed)
            mask, gradient = policy.sample(draw.risk, rng)
            earned = _session_reward(draw, mask, inputs, costs)
            advantages.append(earned - _session_reward(draw, nothing, inputs, costs))  # baseline: the same session with no overbooking
            gradients.append(gradient)
            rewards.append(earned)
        advantage = np.array(advantages)
        centred = advantage - advantage.mean()
        spread = centred.std()
        if spread > 0:
            centred = centred / spread
        step = (centred[:, None] * np.array(gradients)).mean(axis=0)  # REINFORCE: E[advantage x grad log pi]
        first_moment = 0.9 * first_moment + 0.1 * step  # Adam, ascending the reward
        second_moment = 0.999 * second_moment + 0.001 * step ** 2
        theta = theta + learning_rate * (first_moment / (1 - 0.9 ** iteration)) / (np.sqrt(second_moment / (1 - 0.999 ** iteration)) + 1e-8)
        if iteration % validate_every == 0 or iteration == iterations:
            log(iteration, float(np.mean(rewards)), float(advantage.mean()))

    table = pd.DataFrame(rows)
    table["kept"] = False
    best = int(table["validation_reward"].idxmax())  # the first of the best: the earliest weights that did as well
    table.loc[best, "kept"] = True
    kept = table.loc[best, [f"w_{name}" for name in FEATURES]].to_numpy(dtype=float)
    return LearnedPolicy(kept, scaling, costs.name), table


# ---- evaluation --------------------------------------------------------------------------------------------------
def _cost_columns(costs: Costs) -> dict:
    return {"cost_setting": costs.name, "wait_cost_per_hour": costs.wait_per_hour, "overtime_cost_per_hour": costs.overtime_per_hour}


def _reward_columns(rewards, suffix: str = "") -> dict:
    mean, low, high = experiments.mean_ci(rewards)
    return {f"reward{suffix}": mean, f"reward{suffix}_lo": low, f"reward{suffix}_hi": high}


def evaluate(inputs: experiments.Inputs, learned: dict, n_reps: int = experiments.N_REPS, base_seed: int = experiments.BASE_SEED) -> dict:
    """Judge the learned policies on the evaluation sessions (the same ones S2 to S6 use). `learned` maps Costs -> policy.

    s7_learned_policy     one row per learned policy with every metric, like S2, plus its reward and the best hand-made
                          rule under the same costs (chosen on these same sessions, which favours the hand-made rule)
    s7_reward_by_policy   every policy scored under every cost setting, with a paired difference against the learned one
    s7_matched            like S6: at the same number of double-booked slots per session as the learned policy, the
                          learned placement against risk ranking, random slots, even spacing and the oracle
    s7_probability_error  like S3: the learned policies when the risk they see is biased
    """
    standard = experiments.standard_arms(inputs.risk_pool)
    p3 = list(learned.values())
    frames = experiments.run_arms(standard + p3, inputs, n_reps, base_seed)
    meta = experiments.session_meta(inputs, n_reps, base_seed)
    p0 = frames["P0"]

    by_policy, main = [], []
    for costs, policy in learned.items():
        learned_rewards = reward_series(frames[policy.name], costs)
        standard_rewards = {arm.name: reward_series(frames[arm.name], costs) for arm in standard}
        best_name = max(standard_rewards, key=lambda name: standard_rewards[name].mean())  # first on a tie
        for arm in standard + [policy]:
            frame = frames[arm.name]
            rewards = learned_rewards if arm is policy else standard_rewards[arm.name]
            difference = experiments.paired_ci(rewards, learned_rewards)
            by_policy.append({**meta, **_cost_columns(costs), "policy": arm.name, "family": arm.family,
                              **_reward_columns(rewards),
                              "reward_vs_learned": difference[0], "reward_vs_learned_lo": difference[1], "reward_vs_learned_hi": difference[2],
                              "is_best_standard": arm.name == best_name,
                              "double_slots_mean": float(frame["double_slots"].mean()),
                              "patients_served": float(frame["served"].mean()), "mean_wait_min": experiments.mean_ci(frame["mean_wait"])[0],
                              "overtime_min": float(frame["overtime"].mean()), "idle_min": float(frame["idle"].mean())})
        row = experiments.policy_table(frames, [policy], {"vs_p0": p0}, meta).iloc[0].to_dict()
        against_best = experiments.paired_ci(learned_rewards, standard_rewards[best_name])
        main.append({**row, **_cost_columns(costs), **_reward_columns(learned_rewards),
                     "best_standard_policy": best_name, "best_standard_reward": float(standard_rewards[best_name].mean()),
                     "reward_vs_best_standard": against_best[0], "reward_vs_best_standard_lo": against_best[1],
                     "reward_vs_best_standard_hi": against_best[2],
                     **{f"w_{name}": float(w) for name, w in zip(FEATURES, policy.theta)}})

    # the same number of double-booked slots, chosen five ways
    collected: dict = {}
    for r in range(n_reps):
        draw = experiments.make_draw(r, inputs, base_seed)
        visible = draw.risk
        for costs, policy in learned.items():
            mask = policy.select(draw, visible)
            m = int(mask.sum())
            masks = {"p3_learned": mask, "risk_top_m": policies.Ranked("risk").select(draw, visible, m),
                     "random": policies.Ranked("random").select(draw, visible, m),
                     "uniform": policies.Ranked("uniform").select(draw, visible, m),
                     "oracle": policies.Ranked("oracle").select(draw, visible, m)}
            for selector, chosen in masks.items():
                collected.setdefault((costs, selector), []).append(experiments.evaluate_mask(draw, chosen, inputs))
    matched_frames = {key: pd.DataFrame(rows, columns=experiments.METRICS) for key, rows in collected.items()}
    matched = []
    for costs in learned:
        learned_rewards = reward_series(matched_frames[(costs, "p3_learned")], costs)
        for selector in SELECTORS:
            frame = matched_frames[(costs, selector)]
            rewards = reward_series(frame, costs)
            difference = experiments.paired_ci(rewards, learned_rewards)
            matched.append({**meta, "scheme": "matched_to_p3", "setting": costs.name, "selector": selector,
                            "wait_cost_per_hour": costs.wait_per_hour, "overtime_cost_per_hour": costs.overtime_per_hour,
                            **experiments.summary_row(frame, {"vs_p0": p0, "vs_random": matched_frames[(costs, "random")],
                                                              "vs_uniform": matched_frames[(costs, "uniform")]}),
                            **_reward_columns(rewards),
                            "reward_vs_learned": difference[0], "reward_vs_learned_lo": difference[1], "reward_vs_learned_hi": difference[2]})

    shifted = []
    for shift in experiments.SHIFTS:
        shifted_frames = frames if shift == 0.0 else experiments.run_arms(p3, inputs, n_reps, base_seed, shift=shift)
        for costs, policy in learned.items():
            frame = shifted_frames[policy.name]
            shifted.append({**meta, "risk_shift": shift, "policy": policy.name, **_cost_columns(costs),
                            **experiments.summary_row(frame, {"vs_p0": p0, "vs_unshifted": frames[policy.name]}),
                            **_reward_columns(reward_series(frame, costs))})

    return {"s7_learned_policy": pd.DataFrame(main), "s7_reward_by_policy": pd.DataFrame(by_policy),
            "s7_matched": pd.DataFrame(matched), "s7_probability_error": pd.DataFrame(shifted)}


# ---- figure --------------------------------------------------------------------------------------------------------
def figure(by_policy: pd.DataFrame, path) -> None:
    """The S2 trade-off picture with the learned policies added: where does each one land?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml.src.plotting import INK_SECONDARY, SURFACE

    styles = {**experiments.FAMILY_STYLE, "P3": ("#7e57c2", "*", "P3 learned (one per cost of delay)")}
    points = by_policy.drop_duplicates("policy")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.16, wspace=0.18)
    experiments._header(fig, "Where the learned policies land among the hand-made rules",
                        "Each point is one policy, averaged over the same simulated sessions. A star is a policy learned for one stated cost of\n"
                        "waiting and overtime (light, moderate, heavy): a higher cost of delay makes the learner double-book fewer slots.",
                        "The learned policy sees the same predicted risk as P2. Numbers: results/s7_reward_by_policy.csv and results/s7_learned_policy.csv")
    for ax, (column, label) in zip(axes, (("mean_wait_min", "Mean wait of attending patients (minutes)"),
                                          ("overtime_min", "Doctor overtime past the session (minutes)"))):
        experiments._style(ax)
        for family, (color, marker, name) in styles.items():
            part = points[points["family"] == family]
            ax.plot(part[column], part["patients_served"], linestyle="none", marker=marker, markersize=13 if family == "P3" else 7,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=4 if family == "P3" else 3)
            if family == "P3":
                for _, row in part.iterrows():
                    ax.annotate(row["policy"].replace("P3 ", ""), (row[column], row["patients_served"]), xytext=(9, -13),
                                textcoords="offset points", fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel(label, color=INK_SECONDARY, fontsize=9)
    axes[0].set_ylabel("Patients served per session", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.79), ncol=4, fontsize=8.5,
               labelcolor=INK_SECONDARY)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- the run ---------------------------------------------------------------------------------------------------------
def run(results_dir=experiments.RESULTS_DIR, n_reps: int = experiments.N_REPS, base_seed: int = experiments.BASE_SEED,
        repo_root=data.ROOT, prepared_inputs=None, training: dict | None = None) -> dict:
    """Train one policy per cost setting, evaluate them, and write the S7 files. Nothing from S2 to S6 is rewritten."""
    results_dir = Path(results_dir)
    inputs, _, files = experiments.build_inputs(results_dir) if prepared_inputs is None else prepared_inputs
    settings = {**TRAINING, **(training or {})}
    learned, logs = {}, []
    for index, costs in enumerate(COST_SETTINGS):
        policy, log = train(inputs, costs, base_seed=base_seed, index=index, **settings)
        log.insert(0, "cost_setting", costs.name)
        learned[costs] = policy
        logs.append(log)
    tables = evaluate(inputs, learned, n_reps, base_seed)
    tables["s7_training"] = pd.concat(logs, ignore_index=True)

    note = json.dumps({"session_minutes_T": inputs.session_minutes, "slot_minutes_L": inputs.slot_minutes,
                       "n_slots_N": inputs.n_slots, "n_reps": n_reps, "base_seed": base_seed,
                       "algorithm": "REINFORCE with a no-overbooking baseline, Adam", "training": settings,
                       "training_seed_offset": TRAIN_OFFSET, "validation_seed_offset": VALIDATION_OFFSET,
                       "features": list(FEATURES), "initial_weights": list(INITIAL_WEIGHTS),
                       "cost_settings": {c.name: {"wait_per_hour": c.wait_per_hour, "overtime_per_hour": c.overtime_per_hour}
                                         for c in COST_SETTINGS}}, sort_keys=True)
    writer = ResultWriter(results_dir, SCRIPT, base_seed, files, repo_root, note=note)
    for name, table in tables.items():
        writer.table(table, f"{name}.csv")
    path = writer.figure_path("s7_learned_policy.png")
    figure(tables["s7_reward_by_policy"], path)
    writer.record_figure(path)
    return {"inputs": inputs, "learned": learned, **tables}


def main() -> None:
    result = run()
    show = ["policy", "double_slots_mean", "patients_served", "mean_wait_min", "overtime_min", "reward", "best_standard_policy",
            "reward_vs_best_standard", "reward_vs_best_standard_lo", "reward_vs_best_standard_hi"]
    print("S7 learned policies (evaluation sessions):")
    print(result["s7_learned_policy"][show].round(3).to_string(index=False))
    weights = ["policy"] + [f"w_{name}" for name in FEATURES]
    print("\nWeights:")
    print(result["s7_learned_policy"][weights].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
