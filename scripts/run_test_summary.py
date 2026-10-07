"""Run the whole test suite on SQLite and MySQL and write results/test_summary.csv and results/test_summary.txt.

    python -m scripts.run_test_summary            # pytest --engine both
    python -m scripts.run_test_summary --engine sqlite

The report's "tests run, passed" table comes from these files, never from a number typed by hand. Each file gets a
results/manifest.json entry like every other result. The script exits with pytest's exit code, so a failing suite is
a failing script (the files are still written, so the failure is on record).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["group", "name", "tests", "passed", "failed", "skipped"]


def _tally(group: str, name: str, outcomes: list) -> dict:
    return {"group": group, "name": name, "tests": len(outcomes), "passed": outcomes.count("passed"),
            "failed": outcomes.count("failed"), "skipped": outcomes.count("skipped")}


def parse_junit(xml_text: str) -> list:
    """Counts for the whole run, per database engine (tests parametrised [sqlite] or [mysql]) and per test file."""
    outcomes = []  # (module file, engine or None, outcome)
    for case in ET.fromstring(xml_text).iter("testcase"):
        module = case.get("classname", "").split(".")
        file = "/".join(p for p in module if p[:1].islower() and not p.startswith("Test")) + ".py"
        engine = re.search(r"\[(sqlite|mysql)\]$", case.get("name", ""))
        if case.find("failure") is not None or case.find("error") is not None:
            result = "failed"
        elif case.find("skipped") is not None:
            result = "skipped"
        else:
            result = "passed"
        outcomes.append((file, engine.group(1) if engine else None, result))
    rows = [_tally("total", "all", [o for _, _, o in outcomes])]
    for engine in ("sqlite", "mysql"):
        rows.append(_tally("engine", engine, [o for _, e, o in outcomes if e == engine]))
    for file in sorted({f for f, _, _ in outcomes}):
        rows.append(_tally("module", file, [o for f, _, o in outcomes if f == file]))
    return rows


def write_results(rows: list, command: str, root: Path = ROOT) -> None:
    sys.path.insert(0, str(root))
    from ml.src.manifest import record_result

    csv_path, txt_path = root / "results" / "test_summary.csv", root / "results" / "test_summary.txt"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    by_key = {(r["group"], r["name"]): r for r in rows}
    total = by_key[("total", "all")]
    lines = [f"Test run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Command: {command}",
             f"Total: {total['tests']} tests, {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped"]
    for engine in ("sqlite", "mysql"):
        r = by_key[("engine", engine)]
        lines.append(f"Engine {engine}: {r['tests']} database tests, {r['passed']} passed, {r['failed']} failed")
    lines.append("")
    lines += [f"{r['name']:34s} {r['tests']:4d} tests {r['passed']:4d} passed {r['failed']:3d} failed" for r in rows if r["group"] == "module"]
    txt_path.write_text("\n".join(lines) + "\n")
    for path in (csv_path, txt_path):
        record_result(path, script="scripts/run_test_summary.py", seed=None, repo_root=root,
                      note="seeds are fixed inside the tests; this file records a run, not an experiment")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engine", choices=("sqlite", "mysql", "both"), default="both")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--engine={args.engine}", f"--junitxml={report}"]
        # tells the one test that inspects the PREVIOUS record to stand aside: this run is replacing that record, and without
        # this a single red run would be recorded, fail that test on the next run, be recorded red again, and so on for ever
        code = subprocess.run(command, cwd=ROOT, env={**os.environ, "CAMS_RECORDING_TEST_RUN": "1"}).returncode
        if not report.is_file():
            print("pytest wrote no report; nothing recorded")
            return code or 1
        write_results(parse_junit(report.read_text()), "pytest " + " ".join(command[3:-1]))
    print("wrote results/test_summary.csv and results/test_summary.txt")
    return code


if __name__ == "__main__":
    sys.exit(main())
