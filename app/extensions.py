"""Flask extension instances, created here so modules can import them without importing the app."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
