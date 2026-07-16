"""Cashtag MCP server.

Transport: Streamable HTTP. This is a remote, hosted connector — stdio would make
it a local-only tool and forfeit the point of the exercise.

The one rule this layer enforces: **tools read the store and nothing else.** No
tool in this file calls Reddit, X, Bright Data, StockTwits, or the Claude API.
Every number returned was computed by the worker on a schedule and is sitting in
an indexed table. A tool call is a handful of local queries, so it returns in
milliseconds and cannot fail because a social platform rate-limited us.

Auth: a bearer token in the Authorization header, which is what claude.ai custom
connectors support. The middleware is pure-ASGI rather than Starlette's
BaseHTTPMiddleware on purpose — BaseHTTPMiddleware buffers, which interferes with
streamable HTTP's long-lived responses.
"""

from __future__ import annotations

import logging
import secrets
from datetime import timedelta

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from .buzz import (
    baseline_coverage_days,
    criteria_text,
    is_provisional,
    is_warming_up,
    list_trending,
    provisional_note,
    ticker_stats,
    warmup_note,
)
from .config import settings
from .db import init_db, last_ingest_time, session_scope, utcnow
from .models import (
    GetTickerInput,
    ListTrendingInput,
    TickerDetail,
    TrendingResponse,
    TrendingTicker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("cashtag.server")

mcp = FastMCP("cashtag_mcp")

#: Ingestion older than this means the worker is probably dead. Tools keep
#: answering — stale data with a warning beats an error — but they say so.
STALE_AFTER = timedelta(hours=3)


def _freshness_note() -> tuple[str, list[str]]:
    """Describe how current the store is. Returns (freshness_text, notes)."""
    notes: list[str] = []
    with session_scope() as session:
        last = last_ingest_time(session)

    if last is None:
        return (
            "no data ingested yet",
            [
                "The store is empty. Either the ingestion worker has not run, or no "
                "source credentials are configured. Run `python scripts/seed_demo.py` "
                "for synthetic demo data, or start the worker with `cashtag-worker`."
            ],
        )

    if last.tzinfo is None:
        from datetime import timezone

        last = last.replace(tzinfo=timezone.utc)

    age = utcnow() - last
    minutes = int(age.total_seconds() // 60)
    freshness = f"last ingest {minutes} min ago ({last.isoformat()})"

    if age > STALE_AFTER:
        notes.append(
            f"Data is stale: the last successful ingest was {minutes} minutes ago. "
            "The worker may be down. Numbers below reflect that older snapshot."
        )
    return freshness, notes


def _synthetic_note(rows: list) -> list[str]:
    """Warn loudly when any returned row is seeded demo data.

    Synthetic data that can be mistaken for real signal is worse than no data —
    someone could trade on it. Every path that can return it says so.
    """
    if any(
        getattr(r, "is_synthetic", False) or (isinstance(r, dict) and r.get("is_synthetic"))
        for r in rows
    ):
        return [
            "SYNTHETIC DATA: at least one result is backed by seeded demo records, "
            "not live social media. Do not interpret it as real market signal."
        ]
    return []


@mcp.tool(
    name="buzz_list_trending",
    annotations={
        "title": "List Trending Tickers by Social Buzz",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        # False: reads a local pre-computed store, never a live external API.
        "openWorldHint": False,
    },
)
def buzz_list_trending(params: ListTrendingInput) -> TrendingResponse:
    """List stocks with unusually high social-media mention volume in the last 24 hours.

    Returns tickers currently flagged as "buzzing", sorted by buzz_score descending.
    A ticker is flagged only when BOTH conditions hold: at least 15 mentions in the
    trailing 24h, AND at least 2x its own 14-day trailing daily average. Requiring
    both filters out low-volume tickers with flashy ratios (1 -> 3 mentions/day) and
    high-volume tickers that are always discussed (AAPL every day of its life).

    Data comes from a pre-computed store populated by a background worker polling
    Reddit, X, Instagram, TikTok, and StockTwits. This tool does not hit any source
    API, so it returns fast and is unaffected by upstream outages.

    Args:
        params (ListTrendingInput): Validated input containing:
            - limit (int): Max tickers to return, 1-100 (default 20).

    Returns:
        TrendingResponse with:
        {
            "count": int,                  # Number of tickers returned
            "generated_at": str,           # UTC ISO timestamp of this snapshot
            "criteria": str,               # The exact thresholds applied
            "data_freshness": str,         # How recently ingestion last wrote
            "notes": [str],                # Caveats: synthetic data, stale ingest
            "tickers": [
                {
                    "ticker": str,                 # e.g. "GME"
                    "mention_count_24h": int,
                    "buzz_score": float,           # 24h mentions / 14d daily avg
                    "baseline_daily_avg": float,
                    "sentiment": {
                        "bullish_pct": float|null, # bullish/(bullish+bearish)*100
                        "bearish_pct": float|null, # 100 - bullish_pct
                        "neutral_pct": float|null, # neutral/all_classified*100
                        "bullish_count": int,
                        "bearish_count": int,
                        "neutral_count": int,
                        "unclassified_count": int
                    },
                    "top_source": str,             # reddit|x|instagram|tiktok|stocktwits
                    "sample_post_urls": [str],     # 2-3 links to spot-check
                    "is_synthetic": bool
                }
            ]
        }

        IMPORTANT on the percentages: bullish_pct and bearish_pct sum to 100 (they
        are a directional split over opinionated posts only). neutral_pct uses a
        different denominator — all classified posts — so the three do NOT sum to
        100. That is intended, not a bug. bullish_pct is null when a ticker has no
        opinionated posts at all.

    Examples:
        - "What stocks are buzzing right now?" -> limit=20
        - "Top 5 trending tickers" -> limit=5
        - Don't use when: you want data on ONE specific ticker, whether or not it
          is buzzing -> use buzz_get_ticker instead.

    Error handling:
        Returns an empty ticker list with an explanatory note when the store is
        empty or nothing currently meets the thresholds. Both are normal states,
        not errors — a quiet tape genuinely has nothing buzzing.
    """
    with session_scope() as session:
        rows = list_trending(session, limit=params.limit)
        coverage = baseline_coverage_days(session)

    freshness, notes = _freshness_note()

    # Warm-up must be reported explicitly. An empty list with no explanation reads
    # as "nothing is buzzing" — a confident, wrong answer — when the truth is
    # "we cannot know yet".
    if is_warming_up(coverage):
        notes.append(warmup_note(coverage))
    elif is_provisional(coverage):
        notes.append(provisional_note(coverage))

    notes.extend(_synthetic_note(rows))

    if not rows and not is_warming_up(coverage):
        notes.append(
            "No tickers currently meet both buzz thresholds. This is a normal state "
            "on a quiet tape, not an error."
        )

    return TrendingResponse(
        count=len(rows),
        generated_at=utcnow().isoformat(),
        criteria=criteria_text(),
        data_freshness=freshness,
        notes=notes,
        tickers=[TrendingTicker(**row) for row in rows],
    )


@mcp.tool(
    name="buzz_get_ticker",
    annotations={
        "title": "Get Social Buzz Detail for One Ticker",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def buzz_get_ticker(params: GetTickerInput) -> TickerDetail:
    """Get social-media mention and sentiment detail for a specific ticker.

    Works for ANY tracked ticker, whether or not it is currently buzzing — use this
    when the user names a symbol. `is_buzzing` reports whether it currently clears
    both flagging thresholds.

    Reads only the pre-computed store; does not hit any source API.

    Args:
        params (GetTickerInput): Validated input containing:
            - ticker (str): Symbol with or without '$' (e.g. "TSLA", "$GME", "nvda").
              Normalized to uppercase, '$' stripped.

    Returns:
        TickerDetail with:
        {
            "ticker": str,
            "is_buzzing": bool,            # Meets BOTH thresholds right now
            "mention_count_24h": int,
            "buzz_score": float,
            "baseline_daily_avg": float,
            "sentiment": { ... },          # Same shape as buzz_list_trending
            "source_breakdown": [
                {"source": str, "mention_count": int}   # Most mentions first
            ],
            "trend_7d": [
                {"date": str, "mention_count": int}     # 7 UTC days, oldest first,
                                                        # zero-filled where quiet
            ],
            "sample_post_urls": [str],
            "is_synthetic": bool,
            "notes": [str]
        }

        See buzz_list_trending's docstring for the percentage-denominator caveat:
        bullish_pct + bearish_pct = 100, but neutral_pct is computed separately and
        does not belong to that sum.

    Examples:
        - "How is GME sentiment looking?" -> ticker="GME"
        - "Is anyone talking about $SOFI?" -> ticker="$SOFI"
        - Don't use when: you want the overall trending list -> use buzz_list_trending.

    Error handling:
        A ticker with no mentions returns zeroed counts and a note, not an error —
        "nobody is talking about this" is a real and useful answer. Pydantic rejects
        malformed symbols (non-alphabetic, empty) before this function runs.
    """
    with session_scope() as session:
        stats = ticker_stats(session, params.ticker)

    _, notes = _freshness_note()
    # ticker_stats supplies its own notes (warm-up / provisional). Merge, don't
    # overwrite — clobbering them would drop the warning that the buzz_score is
    # not yet meaningful, which is the whole reason it exists.
    notes = list(stats.get("notes", [])) + notes
    notes.extend(_synthetic_note([stats]))

    if stats["mention_count_24h"] == 0:
        notes.append(
            f"No mentions of {params.ticker} in the last 24 hours across any tracked source. "
            f"If this is unexpected, confirm {params.ticker} is in the tracked universe "
            "(see src/cashtag/tickers.py)."
        )

    stats["notes"] = notes
    return TickerDetail(**stats)


# ---------------------------------------------------------------------------
# Transport, auth, health
# ---------------------------------------------------------------------------


class BearerAuthMiddleware:
    """Pure-ASGI bearer token check.

    Pure ASGI rather than BaseHTTPMiddleware because BaseHTTPMiddleware buffers
    responses, which breaks streamable HTTP's long-lived connections.

    Uses `secrets.compare_digest` rather than `==`: token comparison with `==`
    short-circuits on the first differing byte and leaks the token's prefix to
    anyone who can time the response.
    """

    def __init__(self, app, token: str, exempt_paths: frozenset[str] = frozenset({"/health"})):
        self.app = app
        self.token = token
        self.exempt_paths = exempt_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            return await self.app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        provided = headers.get("authorization", "")

        if not secrets.compare_digest(provided, f"Bearer {self.token}"):
            response = JSONResponse(
                {"error": "unauthorized", "detail": "Valid bearer token required."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    """Unauthenticated liveness + data-state probe.

    Exempt from auth so a platform health check can reach it. It deliberately
    exposes no mention data, no ticker names, and no credentials — only whether
    the process is up and whether the store looks alive.
    """
    from sqlalchemy import func, select

    from .db import Mention

    try:
        with session_scope() as session:
            total = session.execute(select(func.count(Mention.id))).scalar_one()
            unclassified = session.execute(
                select(func.count(Mention.id)).where(Mention.sentiment.is_(None))
            ).scalar_one()
            synthetic = session.execute(
                select(func.count(Mention.id)).where(Mention.is_synthetic.is_(True))
            ).scalar_one()
            last = last_ingest_time(session)
    except Exception as exc:
        logger.exception("Health check failed")
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)

    return JSONResponse(
        {
            "status": "ok",
            "mentions_total": total,
            "mentions_unclassified": unclassified,
            "mentions_synthetic": synthetic,
            "last_ingest": last.isoformat() if last else None,
            "auth": "enabled" if settings.auth_token else "DISABLED",
        }
    )


def build_app():
    """Build the ASGI app with auth wrapped around the MCP transport."""
    init_db()
    app = mcp.streamable_http_app()

    if settings.auth_token:
        app = BearerAuthMiddleware(app, settings.auth_token)
        logger.info("Bearer auth enabled")
    else:
        # Loud, because shipping this open to a public URL is the failure mode
        # that matters. Local dev without a token is fine; a public host is not.
        logger.warning(
            "CASHTAG_AUTH_TOKEN is not set — the server is UNAUTHENTICATED. "
            "Acceptable for local development only. Set a token before deploying."
        )
    return app


def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
