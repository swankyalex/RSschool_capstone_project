"""Nox sessions."""

import nox
from nox.sessions import Session

nox.options.sessions = "black", "mypy", "isort", "flake8"
locations = ("src",)


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.run("black", external=True, *args)


@nox.session(python="3.9")
def isort(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.run("isort", external=True, *args)


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.run("flake8", external=True, *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.run("mypy", external=True, *args)
