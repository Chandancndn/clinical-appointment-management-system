"""Copy app/static to public/static, where Vercel serves static files from (its CDN, not Flask).

Run `python -m scripts.sync_public` after changing anything in app/static, and commit public/. A test fails
if the two folders differ. Locally Flask still serves app/static itself; public/ only matters on Vercel.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "static"
TARGET = ROOT / "public" / "static"


def sync() -> list[str]:
    """Make public/static an exact copy of app/static; returns the files copied (paths below the folder)."""
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE, TARGET)
    return sorted(str(p.relative_to(TARGET)) for p in TARGET.rglob("*") if p.is_file())


if __name__ == "__main__":
    for name in sync():
        print("public/static/" + name)
