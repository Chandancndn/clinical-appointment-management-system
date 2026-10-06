"""Friendly error pages. A 404 never says whether the thing exists but belongs to someone else."""
from __future__ import annotations

from flask import render_template
from werkzeug.exceptions import HTTPException

from .extensions import db

MESSAGES = {
    400: ("Request could not be verified", "The form could not be verified, or it was incomplete. "
                                           "Go back, reload the page and try again."),
    403: ("Not allowed", "Your account does not have access to this page."),
    404: ("Page not found", "We could not find what you were looking for."),
    405: ("Method not allowed", "That action is not available this way."),
    409: ("Conflict", "That conflicts with the current state. Reload and try again."),
}


def init_errors(app) -> None:
    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        title, message = MESSAGES.get(error.code, (error.name, error.description or ""))
        return render_template("errors/error.html", code=error.code, title=title, message=message), error.code

    @app.errorhandler(Exception)
    def unexpected(error: Exception):
        db.session.rollback()
        if app.config.get("TESTING") or app.debug:
            raise error
        app.logger.exception("unhandled error")
        return render_template("errors/error.html", code=500, title="Something went wrong",
                               message="An unexpected error occurred. Nothing was changed."), 500
