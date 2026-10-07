"""Quote numbers out of results/ and remember where each one came from.

Every number in docs/report_assets/ and docs/RESULTS_SUMMARY.md is fetched through Registry.cell() or Registry.ci(),
which reads the cell from a file in results/, formats it, and records one row for docs/numbers_audit.csv:
the number as quoted, the results file, the row selector, the column, the format and the document that shows it.
scripts/check_numbers.py re-reads every row from the files and fails if a value has moved.

This module reads CSV files and nothing else: no model, simulation or app code (a test enforces it).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_COLUMNS = ["id", "number", "results_file", "row_selector", "column", "format", "used_in"]
INTERVAL_JOIN = " to "  # a word, not a dash: intervals such as -0.001 to 0.004 stay readable


def read_table(path: Path) -> list:
    """A CSV as a list of dicts of raw strings (no number parsing, empty cells stay empty)."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def render(raw: str, fmt: str) -> str:
    """The cell as the report quotes it. "{}" keeps text as it is; "{pp}" turns a share into signed percentage points
    (-0.0200 becomes -2.0); every other format is a Python format for a number."""
    if fmt == "{}":
        return raw
    if fmt == "{pp}":
        return f"{float(raw) * 100:+.1f}"
    return fmt.format(float(raw))


class Registry:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.rows: list = []
        self._tables: dict = {}
        self._seen: set = set()

    def table(self, results_file: str) -> list:
        if results_file not in self._tables:
            path = self.root / results_file
            if not path.is_file():
                raise FileNotFoundError(f"{results_file} is missing: run its experiment first")
            self._tables[results_file] = read_table(path)
        return self._tables[results_file]

    def raw(self, results_file: str, where: dict, column: str) -> str:
        rows = self.table(results_file)
        if rows and column not in rows[0]:
            raise KeyError(f"{results_file} has no column {column!r}")
        matches = [r for r in rows if all(r.get(k) == str(v) for k, v in where.items())]
        if len(matches) != 1:
            raise KeyError(f"{results_file}: {where} matches {len(matches)} rows, expected exactly 1")
        return matches[0][column]

    def cell(self, results_file: str, where: dict, column: str, fmt: str = "{:.3f}", used_in: str = "") -> str:
        text = render(self.raw(results_file, where, column), fmt)
        key = (text, results_file, json.dumps(where, sort_keys=True), column, fmt, used_in)
        if key not in self._seen:
            self._seen.add(key)
            self.rows.append({"id": f"N{len(self.rows) + 1:04d}", "number": text, "results_file": results_file,
                              "row_selector": json.dumps(where, sort_keys=True), "column": column, "format": fmt,
                              "used_in": used_in})
        return text

    def ci(self, results_file: str, where: dict, column: str, fmt: str = "{:.3f}", used_in: str = "",
           lo: str = None, hi: str = None) -> str:
        """'estimate (low to high)', each of the three recorded."""
        parts = [self.cell(results_file, where, name, fmt, used_in)
                 for name in (column, lo or f"{column}_lo", hi or f"{column}_hi")]
        return f"{parts[0]} ({parts[1]}{INTERVAL_JOIN}{parts[2]})"

    def write_audit(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
