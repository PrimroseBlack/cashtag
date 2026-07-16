"""Tests for database URL normalization.

Regression tests for a bug that reached a real deploy. `render.yaml` injects
Render's `connectionString` — a bare `postgresql://...` — straight into
CASHTAG_DATABASE_URL. SQLAlchemy resolves a bare `postgresql://` to the psycopg2
driver, but pyproject installs psycopg 3, so the service died at startup with:

    ModuleNotFoundError: No module named 'psycopg2'

Nothing caught it because the whole suite runs on SQLite, so no test ever built
a Postgres engine. These tests close that gap without needing a live database:
`create_engine` resolves and imports the driver eagerly, so it fails on a bad URL
long before anyone tries to connect.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from cashtag.db import _normalize_database_url


class TestNormalizeDatabaseUrl:
    def test_bare_postgresql_url_gets_psycopg3_driver(self):
        # The exact shape Render, Heroku, Fly, and Railway inject.
        assert (
            _normalize_database_url("postgresql://u:p@host:5432/db")
            == "postgresql+psycopg://u:p@host:5432/db"
        )

    def test_legacy_postgres_scheme_is_upgraded(self):
        # `postgres://` is not a SQLAlchemy dialect at all — it raises
        # NoSuchModuleError rather than resolving to a driver.
        assert (
            _normalize_database_url("postgres://u:p@host:5432/db")
            == "postgresql+psycopg://u:p@host:5432/db"
        )

    def test_explicit_psycopg_driver_is_left_alone(self):
        url = "postgresql+psycopg://u:p@host:5432/db"
        assert _normalize_database_url(url) == url

    def test_other_explicit_drivers_are_respected(self):
        # An explicit driver is a deliberate choice; don't silently override it.
        url = "postgresql+asyncpg://u:p@host:5432/db"
        assert _normalize_database_url(url) == url

    def test_sqlite_is_untouched(self):
        for url in ("sqlite:///cashtag.db", "sqlite:////abs/path.db", "sqlite://"):
            assert _normalize_database_url(url) == url

    def test_credentials_and_query_params_survive_rewriting(self):
        # Render's URLs carry ?sslmode=require; mangling the query string would
        # break TLS rather than the import, which is a much slower failure.
        raw = "postgresql://user:p%40ss@host.render.com:5432/cashtag?sslmode=require"
        out = _normalize_database_url(raw)
        assert out.startswith("postgresql+psycopg://")
        assert out.endswith("user:p%40ss@host.render.com:5432/cashtag?sslmode=require")

    def test_only_the_scheme_is_replaced_not_later_occurrences(self):
        # A password or hostname containing the scheme text must not be rewritten.
        raw = "postgresql://u:postgresql://@host/db"
        assert _normalize_database_url(raw) == "postgresql+psycopg://u:postgresql://@host/db"


class TestEngineActuallyBuilds:
    """create_engine resolves and imports the driver eagerly — no server needed."""

    def test_bare_render_style_url_builds_an_engine(self):
        # This is the exact call that raised ModuleNotFoundError in production.
        engine = create_engine(_normalize_database_url("postgresql://u:p@host:5432/db"))
        assert engine.dialect.driver == "psycopg"

    def test_legacy_scheme_builds_an_engine(self):
        engine = create_engine(_normalize_database_url("postgres://u:p@host:5432/db"))
        assert engine.dialect.driver == "psycopg"

    def test_unnormalized_bare_url_still_fails(self):
        # Pins the underlying SQLAlchemy behaviour. If a future version stops
        # defaulting to psycopg2, this fails and tells us the workaround is
        # obsolete — rather than leaving it in place forever as cargo cult.
        with pytest.raises(ModuleNotFoundError, match="psycopg2"):
            create_engine("postgresql://u:p@host:5432/db")

    def test_sqlite_still_builds(self):
        engine = create_engine(_normalize_database_url("sqlite://"))
        assert engine.dialect.name == "sqlite"
