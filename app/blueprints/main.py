"""The front door: send each visitor to the right place."""
from __future__ import annotations

from flask import Blueprint, g, redirect, url_for

from ..security import home_for

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    if g.user is None:
        return redirect(url_for("auth.login"))
    return redirect(home_for(g.user))
