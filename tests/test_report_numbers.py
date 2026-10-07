"""M8: the numbers audit, the report assets and the results summary (CLAUDE.md hard rule 3).

  * every number the report quotes comes out of a file in results/ through scripts/reportlib.py, which records it in
    docs/numbers_audit.csv; scripts/check_numbers.py re-reads each file and fails if a quoted value no longer matches;
  * the assets are built from results/ only: the scripts import no model, simulation or app code;
  * the results summary cannot contain a bare number: a template with one is refused.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# the tables and figures PLAN section 8 lists, by the file names this project gives them
PLANNED_TABLES = {
    "t01_dataset_summary": "E1", "t02_cleaning_log": "E1", "t03_baselines": "E2", "t04_kaggle_models": "E3",
    "t05_model_ladder": "E3", "t06_ablation": "E4", "t07_sms_by_lead_time": "E4", "t08_openml_models": "E5",
    "t09_transfer": "E6", "t10_imbalance": "E7", "t11_calibration": "E8", "t12_deployable": "E9",
    "t13_thresholds": "E10", "t14_risk_bands": "E10", "t15_policy_tradeoffs": "S2", "t16_probability_error": "S3",
    "t17_base_rate": "S4", "t18_service_variability": "S5", "t19_value_of_prediction": "S6",
    "t20_service_time_fit": "S1", "t21_test_summary": "tests",
}
PLANNED_FIGURES = (
    "f01_e1_overall_noshow_rate", "f02_e1_noshow_by_lead_time", "f03_e1_noshow_by_weekday", "f04_e1_noshow_by_age_band",
    "f05_e3_roc", "f06_e3_pr", "f07_e4_ablation", "f08_e6_transfer", "f09_e8_reliability", "f10_e10_thresholds",
    "f11_s2_tradeoffs", "f12_s3_probability_error", "f13_s4_base_rate", "f14_s5_service_variability",
    "f15_s6_value_of_prediction", "f16_s6_paired_differences", "f17_service_time_fit", "f18_service_time_qq",
)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def mini(tmp_path):
    """A tiny results/ folder with one table, and a docs/ folder to quote it in."""
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "results" / "mini.csv").write_text(
        "model,auc,auc_lo,auc_hi,note\n"
        "alpha,0.7321,0.7237,0.7401,first\n"
        "beta,0.6124,0.6018,0.6227,\n")
    return root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Everything built once from the real results/ (figures included), shared by the module."""
    from scripts import make_report_assets

    docs = tmp_path_factory.mktemp("docs")
    make_report_assets.build(ROOT, docs, figures=True)
    return docs


# ---- the registry: every quoted number is looked up in a file and recorded --------------------------------------
def test_a_quoted_number_is_the_formatted_cell_and_is_recorded(mini):
    from scripts.reportlib import Registry

    reg = Registry(mini)
    assert reg.cell("results/mini.csv", {"model": "alpha"}, "auc", "{:.3f}", used_in="docs/x.md") == "0.732"
    assert reg.cell("results/mini.csv", {"model": "beta"}, "auc", "{:.1%}", used_in="docs/x.md") == "61.2%"
    assert reg.cell("results/mini.csv", {"model": "alpha"}, "note", "{}", used_in="docs/x.md") == "first"
    assert [r["number"] for r in reg.rows] == ["0.732", "61.2%", "first"]
    row = reg.rows[0]
    assert row["results_file"] == "results/mini.csv" and row["column"] == "auc" and row["used_in"] == "docs/x.md"
    assert json.loads(row["row_selector"]) == {"model": "alpha"}


def test_an_interval_records_the_estimate_and_both_ends(mini):
    from scripts.reportlib import Registry

    reg = Registry(mini)
    assert reg.ci("results/mini.csv", {"model": "alpha"}, "auc", "{:.3f}", used_in="docs/x.md") == "0.732 (0.724 to 0.740)"
    assert [r["column"] for r in reg.rows] == ["auc", "auc_lo", "auc_hi"]


def test_a_selector_must_match_exactly_one_row_and_a_real_column(mini):
    from scripts.reportlib import Registry

    reg = Registry(mini)
    with pytest.raises(KeyError):
        reg.cell("results/mini.csv", {"model": "gamma"}, "auc")
    with pytest.raises(KeyError):
        reg.cell("results/mini.csv", {}, "auc")  # two rows match
    with pytest.raises(KeyError):
        reg.cell("results/mini.csv", {"model": "alpha"}, "no_such_column")
    with pytest.raises(FileNotFoundError):
        reg.cell("results/missing.csv", {"model": "alpha"}, "auc")


def test_the_audit_file_has_the_documented_columns(mini):
    from scripts.reportlib import Registry

    reg = Registry(mini)
    reg.cell("results/mini.csv", {"model": "alpha"}, "auc", used_in="docs/x.md")
    reg.write_audit(mini / "docs" / "numbers_audit.csv")
    rows = read_csv(mini / "docs" / "numbers_audit.csv")
    assert list(rows[0]) == ["id", "number", "results_file", "row_selector", "column", "format", "used_in"]


# ---- the checker fails when a quoted value no longer matches ---------------------------------------------------
def audit_for(mini, doc_text="AUC is 0.732 and 0.724 to 0.740."):
    from scripts.reportlib import Registry

    reg = Registry(mini)
    reg.ci("results/mini.csv", {"model": "alpha"}, "auc", "{:.3f}", used_in="docs/note.md")
    (mini / "docs" / "note.md").write_text(doc_text)
    reg.write_audit(mini / "docs" / "numbers_audit.csv")
    return mini / "docs" / "numbers_audit.csv"


def test_the_checker_passes_when_every_number_matches(mini):
    from scripts.check_numbers import check

    assert check(audit_for(mini), mini) == []


def test_the_checker_fails_when_a_result_value_changes(mini):
    from scripts.check_numbers import check

    audit = audit_for(mini)
    path = mini / "results" / "mini.csv"
    path.write_text(path.read_text().replace("0.7321", "0.7410"))
    problems = check(audit, mini)
    assert problems and any("0.732" in p and "0.741" in p for p in problems)


def test_the_checker_fails_when_a_file_row_or_column_is_gone(mini):
    from scripts.check_numbers import check

    audit = audit_for(mini)
    path = mini / "results" / "mini.csv"
    text = path.read_text()
    path.write_text(text.replace("alpha", "renamed"))
    assert any("matches no row" in p for p in check(audit, mini))
    path.write_text(text.replace("auc_lo", "other"))
    assert any("column" in p for p in check(audit, mini))
    path.unlink()
    assert any("missing" in p for p in check(audit, mini))


def test_the_checker_fails_when_the_document_no_longer_shows_the_number(mini):
    from scripts.check_numbers import check

    audit = audit_for(mini, doc_text="AUC is 0.999.")
    problems = check(audit, mini)
    assert problems and all("docs/note.md" in p for p in problems)


def test_a_longer_number_does_not_count_as_showing_a_shorter_one(mini):
    from scripts.check_numbers import check

    # "0.7321" in the text does not show a quoted "0.732"
    audit = audit_for(mini, doc_text="AUC is 0.7321 (0.7241 to 0.7401).")
    assert check(audit, mini)


def test_the_real_audit_matches_the_real_results():
    from scripts.check_numbers import check

    audit = ROOT / "docs" / "numbers_audit.csv"
    assert audit.is_file(), "run python -m scripts.make_report_assets first"
    assert check(audit, ROOT) == []


def test_the_real_audit_has_a_row_for_every_headline_number():
    rows = read_csv(ROOT / "docs" / "numbers_audit.csv")
    quoted = {(r["results_file"], r["column"]) for r in rows}
    for needed in [("results/e3_kaggle_models.csv", "auc_roc"), ("results/e9_deployable.csv", "auc_roc"),
                   ("results/e10_thresholds.csv", "threshold"), ("results/s6_value_of_prediction.csv", "mean_wait_min_vs_uniform"),
                   ("results/dataset_summary.csv", "value"), ("results/test_summary.csv", "passed")]:
        assert needed in quoted, needed
    assert len(rows) >= 300


# ---- the assets come from results/ only ----------------------------------------------------------------------
def imported_roots(path: Path) -> set:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def test_the_report_scripts_import_no_model_simulation_or_app_code():
    forbidden = {"ml", "sim", "app", "db", "sklearn", "simpy", "flask", "sqlalchemy", "imblearn", "joblib"}
    for name in ("reportlib.py", "check_numbers.py", "make_report_assets.py"):
        assert not imported_roots(SCRIPTS / name) & forbidden, name


def test_every_planned_table_and_figure_is_built(built):
    tables = built / "report_assets" / "tables"
    for name in PLANNED_TABLES:
        for suffix in (".csv", ".md"):
            assert (tables / f"{name}{suffix}").stat().st_size > 20, f"{name}{suffix}"
    for name in PLANNED_FIGURES:
        png = built / "report_assets" / "figures" / f"{name}.png"
        data = png.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000, name
    index = (built / "report_assets" / "INDEX.md").read_text()
    for name in list(PLANNED_TABLES) + list(PLANNED_FIGURES):
        assert name in index, f"{name} is not listed in INDEX.md"


def test_tables_are_clean_and_labelled(built):
    for name in PLANNED_TABLES:
        text = (built / "report_assets" / "tables" / f"{name}.md").read_text()
        assert text.startswith("**Table"), name  # numbered caption above the table
        assert "Source:" in text and "results/" in text, name
        with open(built / "report_assets" / "tables" / f"{name}.csv", newline="") as handle:
            cells = [cell for row in csv.reader(handle) for cell in row]
        assert not [c for c in cells if c.strip().lower() in ("", "nan", "null", "inf", "-inf")], f"{name}: empty or NaN cell"
        assert re.search(r"^\|.*\|\n\|[-: |]+\|\n", text, re.M), f"{name} has no Markdown table"
        with open(built / "report_assets" / "tables" / f"{name}.csv", newline="") as handle:
            header = next(csv.reader(handle))
        assert all(h and h.strip() == h and "_" not in h for h in header), f"{name}: headers should be readable words"


def test_rebuilding_gives_identical_tables_summary_and_audit(built, tmp_path):
    from scripts import make_report_assets

    make_report_assets.build(ROOT, tmp_path, figures=False)
    for relative in sorted(p.relative_to(built) for p in built.rglob("*") if p.is_file() and p.suffix != ".png"):
        assert (tmp_path / relative).read_bytes() == (built / relative).read_bytes(), relative


def test_the_committed_docs_are_what_a_fresh_build_produces(tmp_path):
    """No hand-edited number can hide in docs/: the committed copies equal a build from results/."""
    from scripts import make_report_assets

    make_report_assets.build(ROOT, tmp_path, figures=False)
    for relative in ("numbers_audit.csv", "RESULTS_SUMMARY.md"):
        assert (tmp_path / relative).read_bytes() == (ROOT / "docs" / relative).read_bytes(), relative
    for path in (tmp_path / "report_assets").rglob("*"):
        if path.is_file() and path.suffix != ".png":
            committed = ROOT / "docs" / path.relative_to(tmp_path)
            assert committed.read_bytes() == path.read_bytes(), committed


def test_copied_or_repaired_figures_keep_their_pixels_where_the_curves_are(built):
    """ROC, PR and QQ curves were not saved as data, so their images are reused; only the clipped footer is redrawn."""
    import matplotlib.image as mpimg
    import numpy as np

    for source, name in (("e3_roc", "f05_e3_roc"), ("e3_pr", "f06_e3_pr"), ("sim_service_time_qq", "f18_service_time_qq")):
        original = mpimg.imread(ROOT / "results" / "figures" / f"{source}.png")
        made = mpimg.imread(built / "report_assets" / "figures" / f"{name}.png")
        assert made.shape[1] == original.shape[1] and made.shape[0] >= original.shape[0] - 5
        top = int(original.shape[0] * 0.85)  # everything above the footer strip is untouched
        assert np.allclose(made[:top], original[:top], atol=1e-3), name


# ---- the summary: no bare numbers, every experiment present --------------------------------------------------
def test_a_bare_number_in_the_summary_template_is_refused(mini):
    from scripts.make_report_assets import BareNumber, render_summary
    from scripts.reportlib import Registry

    reg = Registry(mini)
    with pytest.raises(BareNumber):
        render_summary("The model reached 0.73 AUC.", reg)
    with pytest.raises(BareNumber):
        render_summary("About 20% never came.", reg)


def test_labels_and_quoted_cells_are_allowed_in_the_summary_template(mini):
    from scripts.make_report_assets import render_summary
    from scripts.reportlib import Registry

    reg = Registry(mini)
    text = render_summary("E3 and S6 use P2 at p90 or k = 2 (Chapter 8): AUC {{ q('mini', {'model': 'alpha'}, 'auc', 3) }}"
                          " ({{ lit('0.3') }}).", reg)
    assert text == "E3 and S6 use P2 at p90 or k = 2 (Chapter 8): AUC 0.732 (0.3)."
    assert [r["number"] for r in reg.rows] == ["0.732"]


def test_the_summary_covers_every_experiment_with_a_chapter_and_the_limitations(built):
    text = (built / "RESULTS_SUMMARY.md").read_text()
    for experiment in [f"E{i}" for i in range(1, 11)] + [f"S{i}" for i in range(1, 7)]:
        section = re.search(rf"^### {experiment}\b.*?(?=^### |^## |\Z)", text, re.M | re.S)
        assert section, f"no section for {experiment}"
        assert re.search(r"^Report chapter: .*Chapter \d+", section.group(0), re.M), f"{experiment}: which chapter?"
        assert len(section.group(0).split()) > 40, f"{experiment}: too short to say anything"
    lowered = text.lower()
    for needed in ("brazil", "china", "assumed", "advisory", "age and sex", "uncalibrated", "preprint"):
        assert needed in lowered, f"limitations should mention: {needed}"


def test_the_summary_never_says_the_model_beats_a_comparison_it_does_not(built):
    """The S6 paragraph is conditional on the data: it is written from the paired intervals, not from hope."""
    text = (built / "RESULTS_SUMMARY.md").read_text()
    s6 = re.search(r"^### S6\b.*?(?=^### |^## |\Z)", text, re.M | re.S).group(0)
    assert "uniform" in s6 and "random" in s6 and "oracle" in s6
    assert "patients served" in s6.lower()


# ---- the test-summary script ---------------------------------------------------------------------------------
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="5">
<testcase classname="tests.test_a" name="test_x[sqlite]"/>
<testcase classname="tests.test_a" name="test_x[mysql]"/>
<testcase classname="tests.test_a" name="test_y"><failure message="boom"/></testcase>
<testcase classname="tests.test_b" name="test_z[sqlite]"/>
<testcase classname="tests.test_b" name="test_w"><skipped message="n/a"/></testcase>
</testsuite></testsuites>"""


def test_the_junit_report_is_summarised_in_total_by_engine_and_by_module():
    from scripts.run_test_summary import parse_junit

    rows = {(r["group"], r["name"]): r for r in parse_junit(JUNIT)}
    assert rows[("total", "all")] == {"group": "total", "name": "all", "tests": 5, "passed": 3, "failed": 1, "skipped": 1}
    assert rows[("engine", "sqlite")]["tests"] == 2 and rows[("engine", "mysql")]["passed"] == 1
    assert rows[("module", "tests/test_a.py")]["failed"] == 1
    assert rows[("module", "tests/test_b.py")]["skipped"] == 1


def test_the_recorded_test_run_is_all_green_on_both_engines():
    import os

    if os.environ.get("CAMS_RECORDING_TEST_RUN"):
        pytest.skip("scripts/run_test_summary.py is replacing the record this test inspects")
    rows = {(r["group"], r["name"]): r for r in read_csv(ROOT / "results" / "test_summary.csv")}
    total = rows[("total", "all")]
    assert int(total["failed"]) == 0 and int(total["tests"]) == int(total["passed"]) + int(total["skipped"])
    assert int(rows[("engine", "sqlite")]["tests"]) > 0 and int(rows[("engine", "mysql")]["tests"]) > 0
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    assert "results/test_summary.csv" in manifest and "results/test_summary.txt" in manifest
