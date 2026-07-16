"""Scheduled ingestion + classification worker.

Runs as a separate process from the MCP server. That separation is the core
architectural claim of this project: **the MCP tools never touch a source API.**
This worker does all the slow, flaky, rate-limited, billable work on a schedule
and writes results to the store; the tools only read what is already there. A
tool call is a single indexed query against local data, so it answers in
milliseconds regardless of whether Reddit is having a bad day.

Cadence follows the market, not the clock: every 15 minutes while US equities are
open, hourly otherwise. Off-hours chatter is real but slow, and polling it at
market-hours cadence just spends X budget on the same tweets.
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, time as dtime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .classify import classify_pending
from .config import (
    MARKET_CLOSE,
    MARKET_OPEN,
    MARKET_TZ,
    POLL_INTERVAL_MARKET_HOURS_MIN,
    POLL_INTERVAL_OFF_HOURS_MIN,
    settings,
)
from .db import Mention, get_sessionmaker, init_db, session_scope
from .models import Sentiment
from .sources import InstagramSource, RedditSource, StockTwitsSource, TikTokSource, XSource
from .sources.base import RawPost
from .tickers import extract_tickers, load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("cashtag.worker")


def is_market_hours(now: datetime | None = None) -> bool:
    """Whether US equity markets are open right now (weekday + session window).

    Deliberately ignores exchange holidays. A holiday means we poll a quiet tape
    at 15-minute cadence for one day — the cost is a rounding error, and a
    holiday calendar is a dependency plus an annual maintenance obligation.
    Revisit if X spend becomes material.
    """
    now = now or datetime.now(MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    local = now.astimezone(MARKET_TZ)

    if local.weekday() >= 5:  # Saturday, Sunday
        return False

    open_t = dtime(*MARKET_OPEN)
    close_t = dtime(*MARKET_CLOSE)
    return open_t <= local.time() <= close_t


def store_posts(
    session,
    posts: list[RawPost],
    author_tags: dict[tuple[str, str], Sentiment] | None = None,
    is_synthetic: bool = False,
) -> int:
    """Extract tickers from posts and persist one row per (post, ticker).

    Returns the number of NEW rows written. Duplicates are skipped silently —
    every poll re-reads an overlapping window on purpose (a gap loses data
    permanently; an overlap costs nothing), so duplicate suppression is a normal
    hot path, not an error condition.
    """
    author_tags = author_tags or {}
    universe = load_universe()
    written = 0

    for post in posts:
        tickers = extract_tickers(post.text, universe)
        if not tickers:
            continue

        author_tag = author_tags.get((post.source.value, post.source_id))

        for ticker in tickers:
            mention = Mention(
                ticker=ticker,
                source=post.source.value,
                source_id=post.source_id,
                subsource=post.subsource,
                author=post.author,
                text=post.text,
                url=post.url,
                engagement=post.engagement,
                created_utc=post.created_utc,
                sentiment=None,
                author_sentiment=author_tag.value if author_tag else None,
                is_synthetic=is_synthetic,
            )
            # Savepoint per row: a duplicate must roll back only that INSERT, not
            # the whole batch. Without this, one dupe discards every post in the tick.
            try:
                with session.begin_nested():
                    session.add(mention)
                written += 1
            except IntegrityError:
                pass  # Already have this (source, post, ticker). Expected.

    return written


def build_sources(session_factory=None):
    """Instantiate every source adapter.

    X and StockTwits need a ticker list because they query per-symbol; Reddit and
    the Bright Data sources pull feeds and filter afterwards. That asymmetry is
    the access model showing through, not an inconsistency.
    """
    universe = sorted(load_universe())
    session_factory = session_factory or get_sessionmaker()
    return [
        RedditSource(),
        XSource(tickers=universe, session_factory=session_factory),
        StockTwitsSource(tickers=universe),
        InstagramSource(),
        TikTokSource(),
    ]


def ingest_once() -> dict[str, int]:
    """One full ingestion pass across every enabled source.

    Each source is isolated: an exception in one is logged and the rest still run.
    A connector that loses all five sources because Instagram changed a JSON key
    is not a connector.
    """
    results: dict[str, int] = {}
    sources = build_sources()

    for source in sources:
        if not source.enabled:
            logger.info("Skipping %s: %s", source.name.value, source.status())
            results[source.name.value] = 0
            continue

        try:
            author_tags: dict[tuple[str, str], Sentiment] = {}

            if isinstance(source, StockTwitsSource):
                # Only source with author self-tags; keep them for classifier eval.
                pairs = source.fetch_with_tags()
                posts = [p for p, _ in pairs]
                author_tags = {
                    (p.source.value, p.source_id): tag for p, tag in pairs if tag is not None
                }
            else:
                posts = source.fetch()

            with session_scope() as session:
                written = store_posts(session, posts, author_tags=author_tags)
            results[source.name.value] = written
            logger.info("%s: %d new mention rows", source.name.value, written)

        except Exception as exc:
            logger.exception("Source %s failed: %s", source.name.value, exc)
            results[source.name.value] = 0

    return results


def classify_once() -> int:
    """One classification pass over the unlabelled backlog."""
    if not settings.classifier_configured:
        logger.info("Skipping classification: ANTHROPIC_API_KEY not set")
        return 0
    try:
        with session_scope() as session:
            return classify_pending(session)
    except Exception as exc:
        logger.exception("Classification pass failed: %s", exc)
        return 0


def tick() -> None:
    """One full cycle: ingest, then classify what was ingested."""
    phase = "market hours" if is_market_hours() else "off hours"
    logger.info("--- tick start (%s) ---", phase)
    ingested = ingest_once()
    classified = classify_once()
    logger.info(
        "--- tick done: %d new rows across sources, %d rows classified ---",
        sum(ingested.values()),
        classified,
    )


def _reschedule(scheduler: BackgroundScheduler) -> None:
    """Retarget the poll interval as the market opens and closes.

    APScheduler cannot express "15 min during RTH, 60 min otherwise" in one
    trigger, so a cheap supervisor job re-points the interval every 5 minutes.
    """
    want = POLL_INTERVAL_MARKET_HOURS_MIN if is_market_hours() else POLL_INTERVAL_OFF_HOURS_MIN
    job = scheduler.get_job("ingest")
    current = getattr(job.trigger, "interval", None)
    current_min = int(current.total_seconds() // 60) if current else None

    if current_min != want:
        scheduler.reschedule_job("ingest", trigger="interval", minutes=want)
        logger.info(
            "Poll interval -> %d min (%s)",
            want,
            "market hours" if is_market_hours() else "off hours",
        )


def print_status() -> None:
    """Log each source's configuration state at startup.

    Cheaper than discovering at 3am that a source silently never ran.
    """
    logger.info("Source status:")
    for source in build_sources():
        logger.info("  %-12s %s", source.name.value, source.status())
    logger.info(
        "  %-12s %s",
        "classifier",
        f"enabled ({settings.classifier_configured and 'claude-haiku-4-5' or ''})"
        if settings.classifier_configured
        else "disabled (ANTHROPIC_API_KEY not set)",
    )
    with session_scope() as session:
        total = session.execute(select(Mention.id)).scalars().all()
        logger.info("  %-12s %d mention rows in store", "database", len(total))


def assert_store_not_seeded() -> None:
    """Refuse to ingest real data into a store holding synthetic rows.

    Mixing the two silently corrupts the methodology. The seeder writes a
    fabricated 14-day baseline (GME at 8 mentions/day, etc). If the worker then
    adds real GME mentions to that same store, every real buzz_score is computed
    against an invented denominator — a real spike measured against a fake
    normal. Nothing errors; the numbers are just wrong, and they look fine.

    scripts/seed_demo.py already guards the other direction (it refuses to seed
    over real rows). This closes the loop.

    Failing at startup rather than warning per-tick: a warning in a hosted
    worker's log is a warning nobody reads.
    """
    with session_scope() as session:
        synthetic = session.execute(
            select(func.count(Mention.id)).where(Mention.is_synthetic.is_(True))
        ).scalar_one()

    if synthetic:
        logger.error(
            "Refusing to start: this store holds %d SYNTHETIC rows from scripts/seed_demo.py.\n"
            "Ingesting real mentions here would score them against a fabricated baseline and "
            "silently corrupt every buzz_score.\n"
            "\n"
            "Pick one:\n"
            "  - Point the worker at a clean database (recommended — keep the seeded demo "
            "store around for recording demos):\n"
            "      CASHTAG_DATABASE_URL=sqlite:///live.db python -m cashtag.worker\n"
            "  - Or wipe this store completely and start collecting real data in it:\n"
            '      python -c "from cashtag.db import reset_db; reset_db()"\n'
            "\n"
            "Note: `python scripts/seed_demo.py --reset` will NOT help — it drops the tables "
            "and re-seeds them, leaving synthetic data behind again.",
            synthetic,
        )
        sys.exit(1)


def main() -> None:
    """Run the worker until interrupted."""
    init_db()
    logger.info("Cashtag worker starting")
    assert_store_not_seeded()
    print_status()

    scheduler = BackgroundScheduler(timezone=timezone.utc)
    scheduler.add_job(
        tick,
        trigger="interval",
        minutes=POLL_INTERVAL_MARKET_HOURS_MIN
        if is_market_hours()
        else POLL_INTERVAL_OFF_HOURS_MIN,
        id="ingest",
        # A slow tick must not stack behind the next one; skip instead of piling up.
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(_reschedule, trigger="interval", minutes=5, args=[scheduler], id="reschedule")
    scheduler.start()

    tick()  # Run once immediately rather than waiting out the first interval.

    def shutdown(signum, frame):
        logger.info("Shutting down worker")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.pause()


if __name__ == "__main__":
    main()
