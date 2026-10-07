"""Fail if any number the report quotes no longer matches the results file it came from (CLAUDE.md hard rule 3).

    python -m scripts.check_numbers            # the audit table against results/ and the documents that show each number
    python -m scripts.check_numbers --fresh    # also: docs/ must equal a fresh build from results/ (no hand-edited numbers)

For every row of docs/numbers_audit.csv it re-reads the results file, finds the cell, formats it the recorded way and
compares it with the number as quoted. It also checks that the document named in `used_in` still shows that number.
Run it before every submission. The exit code is 1 if anything is off.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

from .reportlib import ROOT, read_table, render


def shows(document: str, number: str) -> bool:
    """True if `number` appears in the text as a whole number (0.7321 does not show 0.732)."""
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", document) is not None


def check(audit_path, root=ROOT) -> list:
    """Every problem found, as readable lines; an empty list means every quoted number matches."""
    root = Path(root)
    problems = []
    with open(audit_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    tables, documents = {}, {}
    for row in rows:
        label = f"{row['id']} {row['number']!r} ({row['results_file']}, {row['row_selector']}, {row['column']})"
        try:
            if row["results_file"] not in tables:
                path = root / row["results_file"]
                if not path.is_file():
                    raise FileNotFoundError(f"results file missing: {row['results_file']}")
                tables[row["results_file"]] = read_table(path)
            table = tables[row["results_file"]]
            if table and row["column"] not in table[0]:
                raise KeyError(f"column {row['column']!r} not in {row['results_file']}")
            where = json.loads(row["row_selector"])
            matches = [r for r in table if all(r.get(k) == str(v) for k, v in where.items())]
            if len(matches) != 1:
                raise KeyError(f"selector {where} matches no row" if not matches else f"selector {where} matches {len(matches)} rows")
            current = render(matches[0][row["column"]], row["format"])
            if current != row["number"]:
                problems.append(f"{label}: quoted {row['number']} but the file now gives {current}")
        except (OSError, KeyError, ValueError) as error:
            problems.append(f"{label}: {error.args[0] if error.args else error}")
        document = row["used_in"]
        if document:
            if document not in documents:
                path = root / document
                documents[document] = path.read_text() if path.is_file() else None
            if documents[document] is None:
                problems.append(f"{label}: the document {document} is missing")
            elif not shows(documents[document], row["number"]):
                problems.append(f"{label}: {document} no longer shows this number")
    return problems


def check_fresh_build(root=ROOT) -> list:
    """docs/ must be exactly what scripts/make_report_assets.py builds from results/ right now."""
    from . import make_report_assets

    root = Path(root)
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        make_report_assets.build(root, Path(tmp), figures=False)
        for built in sorted(p for p in Path(tmp).rglob("*") if p.is_file()):
            committed = root / "docs" / built.relative_to(tmp)
            if not committed.is_file():
                problems.append(f"{committed.relative_to(root)} is missing: run python -m scripts.make_report_assets")
            elif committed.read_bytes() != built.read_bytes():
                problems.append(f"{committed.relative_to(root)} differs from a fresh build (edited by hand, or results changed)")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit", default=str(ROOT / "docs" / "numbers_audit.csv"))
    parser.add_argument("--fresh", action="store_true", help="also compare docs/ with a fresh build")
    args = parser.parse_args(argv)
    problems = check(args.audit)
    if args.fresh:
        problems += check_fresh_build()
    with open(args.audit, newline="") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in {count} audited numbers")
        for line in problems:
            print("  -", line)
        return 1
    print(f"OK: all {count} audited numbers match their results files" + (" and docs/ equals a fresh build" if args.fresh else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
