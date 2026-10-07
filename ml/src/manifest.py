"""results/manifest.json: where every result file came from (CLAUDE.md hard rule 3).

Every script that writes a result file calls record_result() for it. The entry says which script made it,
the random seed, the SHA-256 of each raw data file it read, when it ran and which git commit the code was at.
Re-running a script replaces only its own entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Untracked or modified files here make the code "dirty". results/ and data/ are outputs, not code, and neither is
# ml/artifacts/ (the saved model and its card are written by a script, so they must not make the next run look dirty).
CODE_PATHS = ("ml", "sim", "app", "db", "tests", "scripts", "requirements.txt")
OUTPUT_PATHS = ("ml/artifacts",)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path, root) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:  # outside the repository: keep the full path
        return str(Path(path).resolve())


def _git(root, *args):
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def git_state(root, code_paths=CODE_PATHS):
    """(commit hash, dirty?) for the code. ("unknown", None) outside a repository or before the first commit."""
    commit = _git(root, "rev-parse", "HEAD")
    if not commit:
        return "unknown", None
    paths = [p for p in code_paths if (Path(root) / p).exists()]
    excluded = [f":(exclude){p}" for p in OUTPUT_PATHS]
    status = _git(root, "status", "--porcelain", "--", *paths, *excluded) if paths else ""
    return commit, bool(status)


def record_result(result_path, *, script, seed, data_files=(), repo_root=ROOT, manifest_path=None, note=None) -> dict:
    """Write or replace the manifest entry for `result_path`, which must already exist. Returns the entry."""
    root = Path(repo_root)
    result = Path(result_path)
    if not result.is_file():
        raise FileNotFoundError(f"result file not written yet: {result}")
    hashes = {_relative(path, root): file_sha256(path) for path in data_files}  # raises if a data file is missing
    commit, dirty = git_state(root)
    entry = {
        "script": script,
        "seed": seed,
        "data_files": hashes,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if note:
        entry["note"] = note

    manifest = Path(manifest_path) if manifest_path else root / "results" / "manifest.json"
    stored = json.loads(manifest.read_text()) if manifest.exists() else {}
    stored[_relative(result, root)] = entry

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=manifest.parent, delete=False, suffix=".tmp") as handle:
        json.dump(stored, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(handle.name, 0o644)  # a temp file is created private; the manifest is meant to be shared
    os.replace(handle.name, manifest)  # all or nothing: never a half-written manifest
    return entry
