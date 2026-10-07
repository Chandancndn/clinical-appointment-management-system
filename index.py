"""Vercel entry point: Vercel looks for a Flask object called `app` in a root file like this one.

It builds the same app as `flask --app app run`, with the few settings a hosted, serverless deployment needs.
Everything else (SECRET_KEY, DATABASE_URL, CLINIC_TIMEZONE) comes from Vercel's environment variables.
Run it locally the usual way; this file is only used by Vercel. See "Deploy to Vercel" in README.md.
"""
from werkzeug.middleware.proxy_fix import ProxyFix

from app import create_app

app = create_app({
    "SESSION_COOKIE_SECURE": True,  # Vercel always serves HTTPS
    # A frozen serverless instance loses its database connections silently: test one before using it,
    # and keep the pool small because a hosted MySQL allows few connections.
    "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True, "pool_recycle": 280, "pool_size": 2, "max_overflow": 3},
})
# Behind Vercel's proxy the real visitor address is in X-Forwarded-For. Without this every visitor would look
# like the proxy, and the login throttle (per address) would lock everyone out after a few wrong passwords.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
