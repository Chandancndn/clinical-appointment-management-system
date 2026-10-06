"""results/manifest.json: every script that writes a result records its seed, data hash, date and commit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from ml.src import manifest


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "ml").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "data").mkdir()
    (root / "ml" / "script.py").write_text("print('hi')\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def files(repo):
    raw = repo / "data" / "raw.csv"
    raw.write_text("a,b\n1,2\n")
    result = repo / "results" / "table.csv"
    result.write_text("x\n1\n")
    return raw, result


def read(repo: Path) -> dict:
    return json.loads((repo / "results" / "manifest.json").read_text())


def test_an_entry_records_seed_data_hash_date_and_commit(repo, files):
    raw, result = files
    entry = manifest.record_result(result, script="ml/script.py", seed=42, data_files=[raw], repo_root=repo)

    stored = read(repo)["results/table.csv"]
    assert stored == entry
    assert stored["seed"] == 42 and stored["script"] == "ml/script.py"
    assert stored["data_files"] == {"data/raw.csv": hashlib.sha256(raw.read_bytes()).hexdigest()}
    assert stored["git_commit"] == git(repo, "rev-parse", "HEAD") and len(stored["git_commit"]) == 40
    assert stored["git_dirty"] is False
    assert datetime.fromisoformat(stored["date"]).tzinfo is not None  # an ISO timestamp with timezone


def test_the_entry_says_when_the_code_has_uncommitted_changes(repo, files):
    raw, result = files
    (repo / "ml" / "script.py").write_text("print('changed')\n")
    assert manifest.record_result(result, script="ml/script.py", seed=1, data_files=[raw], repo_root=repo)["git_dirty"] is True
    (repo / "ml" / "new_file.py").write_text("x = 1\n")  # a brand-new untracked source file counts too
    git(repo, "checkout", "--", "ml/script.py")
    assert manifest.record_result(result, script="ml/script.py", seed=1, data_files=[raw], repo_root=repo)["git_dirty"] is True


def test_results_and_raw_data_do_not_make_the_code_look_dirty(repo, files):
    raw, result = files  # untracked files under results/ and data/ are outputs, not code
    assert manifest.record_result(result, script="ml/script.py", seed=1, data_files=[raw], repo_root=repo)["git_dirty"] is False


def test_entries_for_several_results_live_side_by_side_and_rerunning_replaces_only_its_own(repo, files):
    raw, result = files
    other = repo / "results" / "other.csv"
    other.write_text("y\n2\n")
    manifest.record_result(result, script="ml/script.py", seed=1, data_files=[raw], repo_root=repo)
    manifest.record_result(other, script="ml/script.py", seed=2, data_files=[raw], repo_root=repo)
    manifest.record_result(result, script="ml/script.py", seed=3, data_files=[raw], repo_root=repo)
    stored = read(repo)
    assert set(stored) == {"results/table.csv", "results/other.csv"}
    assert stored["results/table.csv"]["seed"] == 3 and stored["results/other.csv"]["seed"] == 2


def test_the_hash_changes_when_the_data_changes(repo, files):
    raw, result = files
    first = manifest.record_result(result, script="s", seed=1, data_files=[raw], repo_root=repo)
    raw.write_text("a,b\n1,3\n")
    second = manifest.record_result(result, script="s", seed=1, data_files=[raw], repo_root=repo)
    assert first["data_files"] != second["data_files"]


def test_outside_a_git_repository_the_commit_is_unknown_not_a_crash(tmp_path):
    result = tmp_path / "r.csv"
    result.write_text("x\n")
    entry = manifest.record_result(result, script="s", seed=1, data_files=[], repo_root=tmp_path,
                                   manifest_path=tmp_path / "manifest.json")
    assert entry["git_commit"] == "unknown" and entry["git_dirty"] is None


def test_a_missing_result_or_data_file_is_an_error(repo, files):
    raw, result = files
    with pytest.raises(FileNotFoundError):
        manifest.record_result(repo / "results" / "not_written.csv", script="s", seed=1, data_files=[raw], repo_root=repo)
    with pytest.raises(FileNotFoundError):
        manifest.record_result(result, script="s", seed=1, data_files=[repo / "data" / "gone.csv"], repo_root=repo)
    assert not (repo / "results" / "manifest.json").exists()  # a failed record leaves no half-written manifest


def test_the_manifest_file_is_readable_by_teammates(repo, files):
    raw, result = files
    manifest.record_result(result, script="s", seed=1, data_files=[raw], repo_root=repo)
    mode = (repo / "results" / "manifest.json").stat().st_mode & 0o777
    assert mode & 0o044 == 0o044, oct(mode)  # group and others can read it


def test_a_seed_is_required():
    with pytest.raises(TypeError):
        manifest.record_result(Path("x"), script="s")  # type: ignore[call-arg]
