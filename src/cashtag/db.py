"""Persistence layer.

SQLite by default; point CASHTAG_DATABASE_URL at Postgres and nothing else
changes. Everything here is plain SQLAlchemy Core-style ORM with no
dialect-specific SQL, which is the whole reason the swap is free.

Hosted Postgres URLs are normalized on the way in — see `_normalize_database_url`.
Do not require callers to hand-craft a driver-qualified URL: on Render, Heroku,
Fly, and Railway the URL is injected by the platform, so there is no human in the
loop to add a suffix to.

Schema note: one row per (post, ticker) pair, not per post. A post saying
"$GME and $AMC to the moon" produces two rows. That is what makes per-ticker
counting a simple GROUP BY instead of a join through an association table, and
it is why the uniqueness constraint includes `ticker`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import settings


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used everywhere; never use naive datetimes."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Mention(Base):
    """A single (post, ticker) observation."""

    __tablename__ = "mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Native ID from the platform. Combined with source+ticker this is what makes
    #: re-polling the same window idempotent instead of double-counting.
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    #: e.g. subreddit name, or an Instagram hashtag. Nullable: not every source has one.
    subsource: Mapped[str | None] = mapped_column(String(64), nullable=True)

    author: Mapped[str | None] = mapped_column(String(128), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    #: Upvotes / likes, normalized across sources. Used only for sample-post ranking.
    engagement: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: When the POST was created, not when we saw it. Every buzz window keys off this.
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    #: NULL until the classification pass labels it. The worker selects on this.
    sentiment: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sentiment_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: The AUTHOR's own sentiment tag, where the platform offers one. Today only
    #: StockTwits does. This is deliberately NOT merged into `sentiment` — keeping
    #: them separate turns StockTwits into a free labelled evaluation set: the
    #: classifier labels those posts blind, and `scripts/eval_classifier.py`
    #: measures its agreement with the human tag. Merging would destroy the only
    #: ground truth in the system.
    author_sentiment: Mapped[str | None] = mapped_column(String(8), nullable=True)

    #: Seeded demo rows. Surfaced through every tool response so synthetic data can
    #: never be mistaken for real signal.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("source", "source_id", "ticker", name="uq_mention_source_post_ticker"),
        Index("ix_mentions_ticker_created", "ticker", "created_utc"),
        Index("ix_mentions_created", "created_utc"),
        # Partial index: the classification worker's hot query is
        # "give me unlabelled rows". Indexing only those keeps it small.
        Index(
            "ix_mentions_unclassified",
            "created_utc",
            sqlite_where=sentiment.is_(None),
            postgresql_where=sentiment.is_(None),
        ),
    )


class ApiSpend(Base):
    """Ledger of paid API reads. Currently only X/Twitter bills per read.

    This exists so the budget cap is enforced against *recorded fact* rather than
    an in-memory counter that resets on every deploy. A process restart must not
    hand the worker a fresh budget.
    """

    __tablename__ = "api_spend"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    reads: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    __table_args__ = (Index("ix_spend_source_time", "source", "occurred_at"),)


class SourceCursor(Base):
    """Per-query pagination cursor.

    The point is cost, not tidiness: X charges per read, so re-fetching tweets we
    already have is money on fire. Storing since_id per query lets each poll ask
    only for what is new.
    """

    __tablename__ = "source_cursors"

    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    query_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    cursor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


_engine = None
_SessionLocal = None


def _normalize_database_url(url: str) -> str:
    """Rewrite a hosted-Postgres URL to name the driver we actually install.

    SQLAlchemy resolves a bare `postgresql://` to `postgresql+psycopg2://` — the
    psycopg2 driver. pyproject installs psycopg 3 (`psycopg[binary]`), so a bare
    URL dies at startup with `ModuleNotFoundError: No module named 'psycopg2'`.

    That is not a theoretical mismatch. Render's `fromDatabase: connectionString`
    injects a bare `postgresql://` URL directly into the environment, and Heroku,
    Fly, and Railway all do the same. There is no human in that path to append a
    `+psycopg` suffix, so requiring one guarantees a failed deploy. Normalizing
    here is the only place that can see every entry point.

    Also handles the legacy `postgres://` scheme (still emitted by some
    providers), which SQLAlchemy refuses outright with NoSuchModuleError rather
    than mapping to a driver at all.

    Args:
        url: Raw value of CASHTAG_DATABASE_URL.

    Returns:
        The same URL with an explicit psycopg3 driver where it was a Postgres
        URL; unchanged for SQLite and for URLs that already name a driver.
    """
    if url.startswith("postgres://"):
        # Legacy scheme — not a SQLAlchemy dialect at all.
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    # Already driver-qualified (postgresql+psycopg://, +asyncpg://, ...) or not
    # Postgres at all. Leave it alone — an explicit choice is the caller's to make.
    return url


def get_engine():
    global _engine
    if _engine is None:
        url = _normalize_database_url(settings.database_url)
        kwargs: dict = {"future": True}
        if url.startswith("sqlite"):
            # check_same_thread=False: the scheduler thread and the ASGI server
            # threads share this engine.
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            # WAL lets the read-only MCP tools query while the worker is writing,
            # instead of tripping over "database is locked".
            with _engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on exception."""
    s = get_sessionmaker()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    """Create tables. Idempotent."""
    Base.metadata.create_all(get_engine())


def reset_db() -> None:
    """Drop and recreate every table. Tests and demo seeding only."""
    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())


def month_to_date_spend(session: Session, source: str, now: datetime | None = None) -> float:
    """Total USD spent on `source` since the first of the current UTC month."""
    now = now or utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = session.execute(
        select(func.coalesce(func.sum(ApiSpend.cost_usd), 0.0)).where(
            ApiSpend.source == source, ApiSpend.occurred_at >= month_start
        )
    ).scalar_one()
    return float(total)


def record_spend(session: Session, source: str, reads: int, cost_per_read: float) -> None:
    """Append to the spend ledger."""
    session.add(
        ApiSpend(source=source, reads=reads, cost_usd=reads * cost_per_read, occurred_at=utcnow())
    )


def last_ingest_time(session: Session) -> datetime | None:
    """Most recent ingestion write, for the freshness field on tool responses."""
    return session.execute(select(func.max(Mention.ingested_at))).scalar_one_or_none()
