"""The admin page "Models and simulation": the saved model card, the risk bands, and the booking-policy simulation.

Everything on it is read from the files the experiments wrote (results/, the model card and the saved figures); the page
computes nothing, shows the same numbers the files hold, is for admins only, and degrades to a message if a file is
missing. Nothing in the booking code reads any of it.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from helpers import text

ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG = {"RISK_ARTIFACTS_DIR": ROOT / "ml" / "artifacts"}
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "s2_policy_tradeoffs.csv").exists(), reason="results/ missing")


def rows(name):
    with open(ROOT / "results" / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_only_admins_may_open_it(scene, world, client, app):
    assert client.get("/admin/research").status_code == 302  # anonymous: to the login page
    for who, expected in ((scene.alice, 403), (scene.doctor_user, 403), (scene.admin, 200)):
        other = app.test_client()
        world.login(other, who)
        assert other.get("/admin/research").status_code == expected, who


def test_the_page_describes_the_model_and_its_bands(scene, world, client):
    world.login(client, scene.admin)
    page = text(client.get("/admin/research"))
    for needed in ("CAMS no-show risk model", "1.0.0", "Vitória", "age", "lead_days", "prev_no_show_rate",
                   "Low", "Medium", "High", "advisory", "never overbooks"):
        assert needed in page, needed


def test_the_numbers_shown_are_the_numbers_in_the_result_files(scene, world, client):
    world.login(client, scene.admin)
    page = text(client.get("/admin/research"))
    holdout = next(r for r in rows("e9_deployable.csv") if r["stage"] == "holdout" and r["model"] == "deployable")
    assert f"{float(holdout['auc_roc']):.3f}" in page and f"{float(holdout['auc_roc_lo']):.3f}" in page
    for band in (r for r in rows("e10_thresholds.csv") if r["kind"] == "band"):
        assert f"{float(band['precision']) * 100:.1f}%" in page, band["label"]
    p0 = next(r for r in rows("s2_policy_tradeoffs.csv") if r["policy"] == "P0")
    assert f"{float(p0['patients_served']):.2f}" in page and f"{float(p0['mean_wait_min']):.1f}" in page
    for policy in rows("s2_policy_tradeoffs.csv"):
        assert policy["policy"] in page


def test_the_page_says_where_the_data_comes_from_and_what_is_assumed(scene, world, client):
    world.login(client, scene.admin)
    page = text(client.get("/admin/research")).lower()
    for needed in ("kaggle", "hangu", "assumed", "results/s2_policy_tradeoffs.csv", "one doctor"):
        assert needed in page, needed


def test_the_figures_are_served_to_admins_by_a_fixed_list_of_names(scene, world, client):
    world.login(client, scene.admin)
    page = text(client.get("/admin/research"))
    for slug in ("policy-tradeoffs", "value-of-prediction", "thresholds", "reliability"):
        response = client.get(f"/admin/research/figures/{slug}.png")
        assert response.status_code == 200 and response.mimetype == "image/png" and response.data[:4] == b"\x89PNG", slug
        assert f"/admin/research/figures/{slug}.png" in page


@pytest.mark.parametrize("name", ["nope", "..", "%2e%2e%2fsecret", "f11_s2_tradeoffs", "../../CLAUDE", "policy-tradeoffs.png.exe"])
def test_nothing_outside_the_list_is_served(name, scene, world, client):
    world.login(client, scene.admin)
    assert client.get(f"/admin/research/figures/{name}.png").status_code == 404


def test_patients_and_doctors_cannot_fetch_the_figures(scene, world, client, app):
    for who in (scene.alice, scene.doctor_user):
        other = app.test_client()
        world.login(other, who)
        assert other.get("/admin/research/figures/thresholds.png").status_code == 403


def test_missing_files_turn_into_messages_not_errors(scene, world, client, app, tmp_path):
    app.config.update(RESULTS_DIR=tmp_path / "none", FIGURES_DIR=tmp_path / "none", RISK_ARTIFACTS_DIR=tmp_path / "none")
    world.login(client, scene.admin)
    response = client.get("/admin/research")
    page = text(response)
    assert response.status_code == 200 and "not found" in page.lower()
    assert client.get("/admin/research/figures/thresholds.png").status_code == 404


def test_the_navigation_links_to_it_for_admins_only(scene, world, client, app):
    world.login(client, scene.admin)
    assert ">Models and simulation<" in text(client.get("/admin/"))
    other = app.test_client()
    world.login(other, scene.alice)
    assert "Models and simulation" not in text(other.get("/patient/"))


def test_the_booking_code_does_not_know_this_page_exists():
    import ast
    from app import services

    names = [n.id for n in ast.walk(ast.parse(Path(services.__file__).read_text())) if isinstance(n, ast.Name)]
    names += [a.name for n in ast.walk(ast.parse(Path(services.__file__).read_text())) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    assert not [n for n in names if "research" in n.lower()]


# ---- S7: the learned policies ------------------------------------------------------------------------------------
needs_s7 = pytest.mark.skipif(not (ROOT / "results" / "s7_learned_policy.csv").exists(), reason="results/s7_learned_policy.csv missing")


@needs_s7
def test_the_learned_policies_are_shown_with_their_stated_costs_and_every_metric(scene, world, client):
    world.login(client, scene.admin)
    page = text(client.get("/admin/research"))
    assert "reinforcement learning" in page.lower() and "results/s7_learned_policy.csv" in page
    for row in rows("s7_learned_policy.csv"):
        assert row["policy"] in page
        for column, digits in (("patients_served", 2), ("mean_wait_min", 1), ("overtime_min", 1), ("double_slots_mean", 2)):
            assert f"{float(row[column]):.{digits}f}" in page, (row["policy"], column)
        assert f"{float(row['reward_vs_best_standard']):+.2f}" in page and row["best_standard_policy"] in page
    lowered = page.lower()
    assert "stated" in lowered and "no right value" in lowered  # the costs are an assumption, and the page says so


@needs_s7
def test_the_learned_policy_figure_comes_from_the_results_folder(scene, world, client):
    world.login(client, scene.admin)
    response = client.get("/admin/research/figures/learned-policy.png")
    assert response.status_code == 200 and response.data[:4] == b"\x89PNG"
    assert "/admin/research/figures/learned-policy.png" in text(client.get("/admin/research"))
