"""The files that let CAMS run on Vercel. None of this changes how the app runs on a laptop; these tests keep the
deployment files honest: the entry point builds the app, static files are where Vercel serves them from, secrets are
never uploaded, and the deployed install stays small (a function bundle is limited to 500 MB)."""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from flask import Flask, request

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ONLY = ("pytest", "matplotlib", "simpy", "imbalanced-learn")


def packages(requirements: Path) -> set[str]:
    names = set()
    for line in requirements.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[=<>~!;\[ ]", line, maxsplit=1)[0].lower())
    return names


@pytest.fixture
def entry(monkeypatch, tmp_path):
    """index.py imported fresh, against a throwaway SQLite file (the app only connects when a request needs it)."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'vercel.sqlite3'}")
    monkeypatch.setenv("SECRET_KEY", "test-only")
    sys.modules.pop("index", None)
    module = importlib.import_module("index")
    yield module
    sys.modules.pop("index", None)


def test_the_entry_point_is_a_flask_app_named_app(entry):
    assert isinstance(entry.app, Flask)
    assert entry.app.config["SESSION_COOKIE_SECURE"] is True  # Vercel is always HTTPS
    options = entry.app.config["SQLALCHEMY_ENGINE_OPTIONS"]
    assert options["pool_pre_ping"] is True and options["pool_size"] <= 5


def test_behind_the_proxy_the_visitors_own_address_is_used(entry):
    entry.app.add_url_rule("/_who", "who", lambda: request.remote_addr)
    seen = entry.app.test_client().get("/_who", headers={"X-Forwarded-For": "203.0.113.9"})
    assert seen.get_data(as_text=True) == "203.0.113.9"  # so the login throttle counts visitors, not Vercel


def test_vercel_serves_the_same_static_files_as_the_app():
    source, mirror = ROOT / "app" / "static", ROOT / "public" / "static"
    files = lambda base: {p.relative_to(base): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    assert files(source) == files(mirror), "run: python -m scripts.sync_public"


def test_vercel_json_points_at_the_entry_point():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert list(config["functions"]) == ["index.py"] and (ROOT / "index.py").exists()


def test_the_deployed_install_leaves_out_the_research_tools():
    deployed, research = packages(ROOT / "requirements.txt"), packages(ROOT / "requirements-research.txt")
    assert not [name for name in RESEARCH_ONLY if name in deployed]
    assert set(RESEARCH_ONLY) <= research
    assert {"flask", "sqlalchemy", "pymysql", "scikit-learn", "pandas", "joblib", "tzdata"} <= deployed
    assert "-r requirements.txt" in (ROOT / "requirements-research.txt").read_text()


def test_secrets_and_raw_data_are_never_uploaded_or_committed():
    lines = (ROOT / ".vercelignore").read_text().splitlines()
    for pattern in (".env", "venv/", "data/raw/", "datasets/"):
        assert pattern in lines, pattern
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored and "data/raw/" in ignored


def test_python_version_is_pinned_for_vercel():
    assert re.fullmatch(r"3\.\d+", (ROOT / ".python-version").read_text().strip())
