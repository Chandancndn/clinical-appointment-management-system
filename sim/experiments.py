"""The simulation experiments S2 to S6.  python -m sim.experiments   (S1 is tests/test_sim.py)

One provider, one half-day session of T minutes with slots of L minutes (N = T / L). Defaults, all computed here:
  T  the seeded app sessions (db/seed_synthetic.py, 09:00 to 13:00)
  L  Hangu's median consultation time rounded up to the next 5 minutes
  consultation times: lognormal or gamma fitted to Hangu in minutes, chosen by AIC (QQ plots saved)
  patient risk: Kaggle holdout bookings scored by the SAVED deployable model, bootstrapped with replacement

COMMON RANDOM NUMBERS: replication r uses numpy seed base_seed + r and draws, in a fixed order, the patients, their
attendance uniforms and their consultation times once; every policy (and every setting of S3 to S5) reuses those draws and only
chooses which slots get a second patient. Differences between policies are therefore PAIRED: the 95% intervals on
"policy minus P0" use the same replications. At least 1,000 replications per setting.

THE TRUTH IN THE SIMULATION IS THE MODEL: a patient's true no-show probability is the model's predicted probability,
and attendance is drawn from it. That is the best case for prediction (the policy knows exactly what the simulated
world knows), so S6 is an upper bound on what a model of this quality can add. The report must say so.

Writes results/s2_policy_tradeoffs.csv, s3_probability_error.csv, s4_base_rate.csv, s5_service_variability.csv,
s6_value_of_prediction.csv, sim_service_time_fit.csv and figures; every file is recorded in results/manifest.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ml.src import data  # noqa: E402
from ml.src.experiment import ResultWriter  # noqa: E402
from ml.src.plotting import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE  # noqa: E402

from . import clinic, inputs as sim_inputs, policies  # noqa: E402

SCRIPT = "sim/experiments.py"
N_REPS = 1000
BASE_SEED = data.DEFAULT_SEED
SHIFTS = (-0.10, -0.05, 0.0, 0.05, 0.10)
BASE_RATE_SCALES = (0.5, 1.0, 1.5)
VARIABILITY_SCALES = (0.5, 1.0, 1.5)

METRICS = ["served", "mean_wait", "overtime", "idle", "both_attend", "double_slots", "both_slots"]
REPORTED = {"served": "patients_served", "mean_wait": "mean_wait_min", "overtime": "overtime_min",
            "idle": "idle_min", "both_attend": "share_sessions_both_attend"}
RESULTS_DIR = data.RESULTS_DIR


@dataclass
class Inputs:
    service_model: sim_inputs.ServiceTimeModel
    risk_pool: np.ndarray
    session_minutes: int
    slot_minutes: int

    @property
    def n_slots(self) -> int:
        if self.session_minutes % self.slot_minutes:
            raise ValueError("the session length must be a whole number of slots")
        return self.session_minutes // self.slot_minutes


# ---- statistics ----------------------------------------------------------------------------------------
def mean_ci(x) -> tuple[float, float, float]:
    """Mean and 95% t-interval over replications, ignoring missing values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    mean = float(x.mean())
    if len(x) < 2 or x.std(ddof=1) == 0:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half


def paired_ci(a, b) -> tuple[float, float, float]:
    """Mean of a - b over replications with a 95% interval; pairs with a missing value are dropped."""
    return mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


# ---- one replication --------------------------------------------------------------------------------------
def make_draw(r: int, inputs: Inputs, base_seed: int, base_rate_scale: float = 1.0, variability: float = 1.0) -> policies.Draw:
    """Replication r's random draws. The uniforms depend only on base_seed + r, so scaling the base rate or the
    service-time variability changes how the same uniforms are turned into patients, never the uniforms themselves."""
    rng = np.random.default_rng(base_seed + r)
    n, pool = inputs.n_slots, inputs.risk_pool
    primary_index, extra_index = rng.integers(0, len(pool), n), rng.integers(0, len(pool), n)
    u_primary, u_extra, v_primary, v_extra, random_scores = (rng.random(n) for _ in range(5))
    model = inputs.service_model
    return policies.Draw(
        risk=np.clip(pool[primary_index] * base_rate_scale, 0.0, 1.0),
        extra_risk=np.clip(pool[extra_index] * base_rate_scale, 0.0, 1.0),
        u_primary=u_primary, u_extra=u_extra,
        service_primary=model.ppf(v_primary, variability), service_extra=model.ppf(v_extra, variability),
        random_scores=random_scores)


def evaluate_mask(draw: policies.Draw, mask: np.ndarray, inputs: Inputs) -> dict:
    session = policies.build_session(draw, mask, inputs.slot_minutes)
    result = clinic.simulate_session(session.arrivals, session.attends, session.services, inputs.session_minutes)
    return {"served": result.served, "mean_wait": result.mean_wait, "overtime": result.overtime, "idle": result.idle,
            "both_attend": float(session.both_slots > 0), "double_slots": session.double_slots,
            "both_slots": session.both_slots}


def run_arms(arms, inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED, shift: float = 0.0,
             base_rate_scale: float = 1.0, variability: float = 1.0) -> dict:
    """Run every policy on the same draws. Returns {policy name: DataFrame, one row per replication}.

    `shift` biases the risk the policy SEES (predicted = true + shift, clipped) while true attendance is unchanged."""
    rows = {arm.name: [] for arm in arms}
    for r in range(n_reps):
        draw = make_draw(r, inputs, base_seed, base_rate_scale, variability)
        visible = np.clip(draw.risk + shift, 0.0, 1.0)
        for arm in arms:
            rows[arm.name].append(evaluate_mask(draw, arm.select(draw, visible), inputs))
    return {name: pd.DataFrame(values, columns=METRICS) for name, values in rows.items()}


# ---- tables --------------------------------------------------------------------------------------------------
def session_meta(inputs: Inputs, n_reps: int, base_seed: int) -> dict:
    return {"session_minutes": inputs.session_minutes, "slot_minutes": inputs.slot_minutes, "n_slots": inputs.n_slots,
            "n_reps": n_reps, "base_seed": base_seed}


def summary_row(frame: pd.DataFrame, references: dict) -> dict:
    """Mean and 95% interval of every reported metric, and paired differences against each reference frame."""
    row = {"double_slots_mean": float(frame["double_slots"].mean()),
           "share_double_slots_both_attend": (float(frame["both_slots"].sum() / frame["double_slots"].sum())
                                              if frame["double_slots"].sum() else math.nan)}
    for metric, label in REPORTED.items():
        row[label], row[f"{label}_lo"], row[f"{label}_hi"] = mean_ci(frame[metric])
    for suffix, reference in references.items():
        for metric, label in REPORTED.items():
            diff = paired_ci(frame[metric], reference[metric])
            row[f"{label}_{suffix}"], row[f"{label}_{suffix}_lo"], row[f"{label}_{suffix}_hi"] = diff
    return row


def policy_table(frames: dict, arms, references: dict, meta: dict) -> pd.DataFrame:
    """One row per policy. `references` maps a column suffix to a reference frame, or to {policy name: frame}."""
    rows = []
    for arm in arms:
        refs = {suffix: (ref[arm.name] if isinstance(ref, dict) else ref) for suffix, ref in references.items()}
        rows.append({**meta, "policy": arm.name, "family": arm.family, "parameter": arm.label,
                     "threshold": arm.threshold if arm.threshold is not None else math.nan,
                     **summary_row(frames[arm.name], refs)})
    return pd.DataFrame(rows)


# ---- S2: policy trade-offs ---------------------------------------------------------------------------------------
def standard_arms(pool) -> list:
    return [policies.NoOverbooking(), *policies.p1_policies(), *policies.p2_policies(pool)]


def s2_policy_tradeoffs(inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED) -> pd.DataFrame:
    arms = standard_arms(inputs.risk_pool)
    frames = run_arms(arms, inputs, n_reps, base_seed)
    return policy_table(frames, arms, {"vs_p0": frames["P0"]}, session_meta(inputs, n_reps, base_seed))


# ---- S3: how much does a biased risk estimate hurt P2? --------------------------------------------------------------
def s3_probability_error(inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED, shifts=SHIFTS) -> pd.DataFrame:
    """The thresholds are fixed from the UNSHIFTED risk distribution, then the risk the policy sees is shifted: a model
    that over- (+) or under- (-) estimates everybody. True attendance does not change."""
    arms = policies.p2_policies(inputs.risk_pool)
    p0 = run_arms([policies.NoOverbooking()], inputs, n_reps, base_seed)["P0"]
    unshifted = run_arms(arms, inputs, n_reps, base_seed, shift=0.0)
    rows = [policy_table({"P0": p0}, [policies.NoOverbooking()], {"vs_p0": p0, "vs_unshifted": p0},
                         {**session_meta(inputs, n_reps, base_seed), "risk_shift": 0.0}).iloc[0].to_dict()]
    for shift in shifts:
        frames = unshifted if shift == 0.0 else run_arms(arms, inputs, n_reps, base_seed, shift=shift)
        table = policy_table(frames, arms, {"vs_p0": p0, "vs_unshifted": unshifted},
                             {**session_meta(inputs, n_reps, base_seed), "risk_shift": shift})
        rows += table.to_dict("records")
    return pd.DataFrame(rows)


# ---- S4: base no-show rate ------------------------------------------------------------------------------------------
def s4_base_rate(inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED, scales=BASE_RATE_SCALES) -> pd.DataFrame:
    """Every patient's risk is multiplied by the scale (for truth AND for what the policy sees: a model recalibrated to
    the new clinic). Percentile thresholds are recomputed on the scaled risks; the fixed 0.3 and 0.5 stay as they are."""
    rows = []
    for scale in scales:
        scaled_pool = np.clip(inputs.risk_pool * scale, 0.0, 1.0)
        arms = standard_arms(scaled_pool)
        frames = run_arms(arms, inputs, n_reps, base_seed, base_rate_scale=scale)
        table = policy_table(frames, arms, {"vs_p0": frames["P0"]},
                             {**session_meta(inputs, n_reps, base_seed), "base_rate_scale": scale,
                              "base_no_show_rate": float(scaled_pool.mean())})
        rows += table.to_dict("records")
    return pd.DataFrame(rows)


# ---- S5: consultation-time variability -------------------------------------------------------------------------------
def s5_service_variability(inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED,
                           scales=VARIABILITY_SCALES) -> pd.DataFrame:
    """The standard deviation of consultation time is multiplied by the scale at the SAME mean."""
    arms = standard_arms(inputs.risk_pool)
    grid = (np.arange(20000) + 0.5) / 20000
    rows = []
    for scale in scales:
        minutes = inputs.service_model.ppf(grid, variability=scale)
        frames = run_arms(arms, inputs, n_reps, base_seed, variability=scale)
        table = policy_table(frames, arms, {"vs_p0": frames["P0"]},
                             {**session_meta(inputs, n_reps, base_seed), "variability_scale": scale,
                              "service_mean_min": float(minutes.mean()), "service_sd_min": float(minutes.std())})
        rows += table.to_dict("records")
    return pd.DataFrame(rows)


# ---- S6: is the model worth having? ---------------------------------------------------------------------------------
SELECTOR_ORDER = ["p2_risk_threshold", "p1_uniform_k", "risk_top_m", "random", "uniform", "oracle"]


def s6_value_of_prediction(inputs: Inputs, n_reps: int = N_REPS, base_seed: int = BASE_SEED) -> pd.DataFrame:
    """Prediction against its alternatives at the SAME number of double-booked slots, session by session.

    matched_to_p2  each P2 threshold decides how many slots to double-book in a session (m); in that same session
                   random scores, even spacing and an oracle that knows who will not attend double-book exactly m slots
    matched_to_p1  each P1 k fixes m = N / k; risk ranking, random scores and the oracle double-book m slots too

    Extras are taken in order from shared draws, so at equal m every selector serves the SAME patients and only the
    placement differs: prediction can show up in waiting, overtime, idle time and collisions, never in patients served.
    """
    pool = inputs.risk_pool
    p0 = policies.NoOverbooking()
    collected: dict = {}
    p0_rows = []

    def put(key, draw, mask):
        collected.setdefault(key, []).append(evaluate_mask(draw, mask, inputs))

    for r in range(n_reps):
        draw = make_draw(r, inputs, base_seed)
        visible = draw.risk  # no bias here: S3 is where the estimate is biased
        p0_rows.append(evaluate_mask(draw, p0.select(draw, visible), inputs))
        for policy in policies.p2_policies(pool):
            mask = policy.select(draw, visible)
            m = int(mask.sum())
            put(("matched_to_p2", policy.label, "p2_risk_threshold"), draw, mask)
            for selector in ("random", "uniform", "oracle"):
                put(("matched_to_p2", policy.label, selector), draw, policies.Ranked(selector).select(draw, visible, m))
        for policy in policies.p1_policies():
            mask = policy.select(draw, visible)
            m = int(mask.sum())
            put(("matched_to_p1", policy.label, "p1_uniform_k"), draw, mask)
            for selector in ("risk", "random", "oracle"):
                put(("matched_to_p1", policy.label, "risk_top_m" if selector == "risk" else selector), draw,
                    policies.Ranked(selector).select(draw, visible, m))

    p0_frame = pd.DataFrame(p0_rows, columns=METRICS)
    frames = {key: pd.DataFrame(rows, columns=METRICS) for key, rows in collected.items()}
    meta = session_meta(inputs, n_reps, base_seed)
    rows = []
    for (scheme, setting, selector), frame in frames.items():
        random_frame = frames[(scheme, setting, "random")]
        row = {**meta, "scheme": scheme, "setting": setting, "selector": selector,
               **summary_row(frame, {"vs_p0": p0_frame, "vs_random": random_frame})}
        row["served_gain_per_double_slot"] = (row["patients_served_vs_p0"] / row["double_slots_mean"]
                                              if row["double_slots_mean"] else math.nan)  # ratio of means, no interval
        rows.append(row)
    table = pd.DataFrame(rows)
    table["_s"] = table["selector"].map(SELECTOR_ORDER.index)
    table["_k"] = table["setting"].map({k: i for i, k in enumerate(dict.fromkeys(table["setting"]))})
    return table.sort_values(["scheme", "_k", "_s"], kind="stable").drop(columns=["_s", "_k"]).reset_index(drop=True)


# ---- figures ---------------------------------------------------------------------------------------------------------
FAMILY_STYLE = {"P0": ("#898881", "s", "P0 no overbooking"), "P1": ("#2a78d6", "o", "P1 uniform (every k-th slot)"),
                "P2": ("#eb6834", "^", "P2 risk threshold")}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.grid(color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _header(fig, title: str, subtitle: str, note: str) -> None:
    fig.text(0.06, 0.955, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.06, 0.895, subtitle, fontsize=9, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)
    fig.text(0.06, 0.02, note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")


def tradeoff_figure(table: pd.DataFrame, path) -> None:
    """Patients served against mean wait and against overtime: the trade-off, one point per policy."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.15, wspace=0.18)
    _header(fig, "What each booking policy trades away",
            "Each point is one policy, averaged over the replications. More patients served costs longer waits and more overtime;\n"
            "no single score is computed, because any weighting of the two would be a choice of the reader.",
            "Simulated session; consultation times from Hangu, no-show risk from the deployable model. Numbers: results/s2_policy_tradeoffs.csv")
    for ax, (column, label) in zip(axes, (("mean_wait_min", "Mean wait of attending patients (minutes)"),
                                          ("overtime_min", "Doctor overtime past the session (minutes)"))):
        _style(ax)
        for family in ("P0", "P1", "P2"):
            color, marker, name = FAMILY_STYLE[family]
            part = table[table["family"] == family]
            ax.plot(part[column], part["patients_served"], linestyle="none", marker=marker, markersize=8, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            for _, r in part.iterrows():
                ax.annotate(r["parameter"] if family != "P0" else "", (r[column], r["patients_served"]), xytext=(5, 4),
                            textcoords="offset points", fontsize=7, color=INK_SECONDARY)
        ax.set_xlabel(label, color=INK_SECONDARY, fontsize=9)
    axes[0].set_ylabel("Patients served per session", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.80), ncol=3, fontsize=8.5,
               labelcolor=INK_SECONDARY)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


S6_STYLE = {"p2_risk_threshold": ("#2a78d6", "o", "Risk threshold (P2)"), "random": ("#eb6834", "s", "Random slots"),
            "uniform": ("#1baf7a", "^", "Evenly spaced slots"), "oracle": ("#eda100", "D", "Oracle (knows who will not come)")}


def value_figure(table: pd.DataFrame, path) -> None:
    """At the same number of double-booked slots per session, how do the ways of choosing them compare?"""
    part = table[table["scheme"] == "matched_to_p2"]
    panels = (("share_sessions_both_attend", "Sessions where both patients of a double-booked slot attend"),
              ("mean_wait_min", "Mean wait of attending patients (minutes)"),
              ("overtime_min", "Doctor overtime (minutes)"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), dpi=200, facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.72, bottom=0.17, wspace=0.28)
    _header(fig, "Is the model worth having? Same number of double-booked slots, different ways of choosing them",
            "Each point is one P2 threshold; every line double-books exactly as many slots as that threshold did in the same session.\n"
            "Patients served are identical across the lines by construction, so the difference shows up only in waiting, overtime and collisions.",
            "The simulated world's true risk IS the model's risk, so this is the most prediction can help. Numbers: results/s6_value_of_prediction.csv")
    for ax, (column, label) in zip(axes, panels):
        _style(ax)
        for selector, (color, marker, name) in S6_STYLE.items():
            rows = part[part["selector"] == selector].sort_values("double_slots_mean")
            ax.plot(rows["double_slots_mean"], rows[column], color=color, marker=marker, markersize=7, linewidth=1.6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
        ax.set_xlabel("Double-booked slots per session (mean)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(label, color=INK_SECONDARY, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.055, 0.80), ncol=4, fontsize=8.5,
               labelcolor=INK_SECONDARY)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- the run ---------------------------------------------------------------------------------------------------------
def build_inputs(results_dir=RESULTS_DIR, writer: ResultWriter | None = None):
    """Fit the consultation times, derive T and L, and score the risk pool. Returns (Inputs, fit, input files)."""
    minutes = data.clean_hangu(data.load_hangu())["service_minutes"].to_numpy()
    fit = sim_inputs.fit_service_times(minutes)
    slot = sim_inputs.default_slot_minutes(float(np.median(minutes)))
    session = sim_inputs.session_minutes_from_seed()
    risk, risk_files = sim_inputs.load_risk_pool()
    return Inputs(fit.model, risk, session, slot), fit, [data.HANGU_PATH, *risk_files]


def fit_table(fit: sim_inputs.FitResult, inputs: Inputs) -> pd.DataFrame:
    table = fit.table()
    other = [f for f in fit.fits if f is not fit.chosen][0]
    reason = (f"{fit.chosen.family} has the lower AIC ({fit.chosen.aic:,.1f} against {other.aic:,.1f}, a difference of "
              f"{other.aic - fit.chosen.aic:,.1f}); QQ plots in results/figures/sim_service_time_qq.png")
    table["choice_reason"] = ["" if not row.chosen else reason for row in table.itertuples()]
    table["median_service_min"] = float(np.median(fit.minutes))
    table["slot_minutes_default"] = inputs.slot_minutes
    table["session_minutes_default"] = inputs.session_minutes
    return table


def run(results_dir=RESULTS_DIR, n_reps: int = N_REPS, base_seed: int = BASE_SEED, repo_root=data.ROOT,
        prepared_inputs=None) -> dict:
    results_dir = Path(results_dir)
    if prepared_inputs is None:
        inputs, fit, files = build_inputs(results_dir)
    else:
        inputs, fit, files = prepared_inputs
    note = json.dumps({"session_minutes_T": inputs.session_minutes, "slot_minutes_L": inputs.slot_minutes,
                       "n_slots_N": inputs.n_slots, "n_reps": n_reps, "base_seed": base_seed,
                       "service_time_family": inputs.service_model.family}, sort_keys=True)
    writer = ResultWriter(results_dir, SCRIPT, base_seed, files, repo_root, note=note)

    fits = fit_table(fit, inputs)
    writer.table(fits, "sim_service_time_fit.csv")
    for name, draw_figure in (("sim_service_time_qq.png", sim_inputs.qq_figure), ("sim_service_time_fit.png", sim_inputs.density_figure)):
        path = writer.figure_path(name)
        draw_figure(fit, path)
        writer.record_figure(path)

    tables = {
        "s2_policy_tradeoffs.csv": s2_policy_tradeoffs(inputs, n_reps, base_seed),
        "s3_probability_error.csv": s3_probability_error(inputs, n_reps, base_seed),
        "s4_base_rate.csv": s4_base_rate(inputs, n_reps, base_seed),
        "s5_service_variability.csv": s5_service_variability(inputs, n_reps, base_seed),
        "s6_value_of_prediction.csv": s6_value_of_prediction(inputs, n_reps, base_seed),
    }
    for name, table in tables.items():
        writer.table(table, name)
    for name, draw_figure, key in (("s2_tradeoffs.png", tradeoff_figure, "s2_policy_tradeoffs.csv"),
                                   ("s6_value_of_prediction.png", value_figure, "s6_value_of_prediction.csv")):
        path = writer.figure_path(name)
        draw_figure(tables[key], path)
        writer.record_figure(path)
    return {"fit": fits, "inputs": inputs, **{name.removesuffix(".csv"): t for name, t in tables.items()}}


def main() -> None:
    result = run()
    inputs = result["inputs"]
    fit = result["fit"]
    print(f"T = {inputs.session_minutes} min, L = {inputs.slot_minutes} min, N = {inputs.n_slots} slots, {N_REPS} replications, seed {BASE_SEED}")
    print(fit[["family", "param_1", "param_2", "log_likelihood", "aic", "chosen"]].round(3).to_string(index=False))
    show = ["policy", "double_slots_mean", "patients_served", "mean_wait_min", "overtime_min", "idle_min", "share_sessions_both_attend"]
    print("\nS2 policy trade-offs:"); print(result["s2_policy_tradeoffs"][show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
