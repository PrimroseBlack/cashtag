"""Buzz scoring and sentiment aggregation.

The pure functions (`compute_buzz_score`, `is_buzzing`, `aggregate_sentiment`)
carry the whole methodology and take no database. That is deliberate: they are
the part that has to be *right*, so they are the part that is trivial to test.
The DB-touching functions below them are plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import (
    BASELINE_DAYS,
    BASELINE_FLOOR,
    BUZZ_MIN_MENTIONS_24H,
    BUZZ_SCORE_THRESHOLD,
    MIN_BASELINE_COVERAGE_DAYS,
    PROVISIONAL_TOLERANCE_DAYS,
    TRAILING_HOURS,
)
from .db import Mention, utcnow
from .models import (
    SentimentSplit,
    Source,
    SourceBreakdown,
    TrendPoint,
)

# ---------------------------------------------------------------------------
# Pure methodology
# ---------------------------------------------------------------------------


def compute_buzz_score(mentions_24h: int, baseline_daily_avg: float) -> float:
    """Ratio of current chatter to normal chatter.

    Args:
        mentions_24h: Mentions in the trailing 24h window.
        baseline_daily_avg: Mean mentions/day over the preceding BASELINE_DAYS.

    Returns:
        mentions_24h / max(baseline_daily_avg, BASELINE_FLOOR), rounded to 2dp.

    The floor matters more than it looks. A ticker with no history has a baseline
    of 0.0, and dividing by it is either a crash or an infinity that pins the
    ticker to the top of the list forever. Flooring the denominator at 0.5
    caps a cold-start ticker's score at 2x its raw count — it can still rank,
    but it cannot dominate on the strength of having no past.
    """
    if mentions_24h <= 0:
        return 0.0
    denominator = max(baseline_daily_avg, BASELINE_FLOOR)
    return round(mentions_24h / denominator, 2)


def is_buzzing(mentions_24h: int, buzz_score: float) -> bool:
    """Whether a ticker clears BOTH flagging thresholds.

    Both conditions are load-bearing and neither works alone:
      - Ratio alone: a ticker going 1 -> 3 mentions/day scores 3.0x and outranks
        real signal. That is noise with a good ratio.
      - Absolute alone: AAPL clears 15 mentions every single day of its life.
        That is volume without news.
    Requiring both means "unusually loud, and loud enough to matter".
    """
    return mentions_24h >= BUZZ_MIN_MENTIONS_24H and buzz_score >= BUZZ_SCORE_THRESHOLD


def aggregate_sentiment(
    bullish: int, bearish: int, neutral: int, unclassified: int = 0
) -> SentimentSplit:
    """Aggregate raw label counts into the reported split.

    Two different denominators, on purpose — see the note in models.py:

      bullish_pct = bullish / (bullish + bearish) * 100   [directional only]
      bearish_pct = 100 - bullish_pct
      neutral_pct = neutral / (bullish + bearish + neutral) * 100   [all labels]

    So the three do NOT sum to 100. bullish_pct/bearish_pct answer "which way are
    the people with an opinion leaning?"; neutral_pct answers "how many even have
    one?". Folding neutral into the directional denominator would mean a ticker
    with 10 bullish, 0 bearish, and 90 neutral posts reports 10% bullish — which
    reads as bearish, when in fact opinion was unanimously bullish.

    Returns null percentages rather than a default when the denominator is zero.
    A ticker with no opinionated posts is genuinely undefined, not 50/50 — and
    50 would be indistinguishable from a real, evenly-split market.
    """
    directional = bullish + bearish
    total_classified = bullish + bearish + neutral

    if directional > 0:
        bullish_pct: float | None = round(bullish / directional * 100, 1)
        bearish_pct: float | None = round(100.0 - bullish_pct, 1)
    else:
        bullish_pct = None
        bearish_pct = None

    neutral_pct = round(neutral / total_classified * 100, 1) if total_classified > 0 else None

    return SentimentSplit(
        bullish_pct=bullish_pct,
        bearish_pct=bearish_pct,
        neutral_pct=neutral_pct,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        unclassified_count=unclassified,
    )


def window_bounds(now: datetime | None = None) -> tuple[datetime, datetime, datetime]:
    """Return (baseline_start, window_start, now).

    The baseline window is [baseline_start, window_start) and the trailing window
    is [window_start, now]. They ABUT but never overlap: the baseline ends exactly
    where the 24h window begins.

    This is the single most consequential line of the methodology. If the trailing
    24h were included in its own baseline, a spike would inflate its own
    denominator and suppress its own score — the detector would be least sensitive
    exactly when something is happening.
    """
    now = now or utcnow()
    window_start = now - timedelta(hours=TRAILING_HOURS)
    baseline_start = window_start - timedelta(days=BASELINE_DAYS)
    return baseline_start, window_start, now


def compute_baseline_avg(baseline_count: int, coverage_days: float = BASELINE_DAYS) -> float:
    """Mean mentions/day across the baseline window.

    Args:
        baseline_count: Mentions observed in the baseline window.
        coverage_days: Days of history ACTUALLY observed. Defaults to the full
            BASELINE_DAYS. Pass the real coverage on a young store: dividing 3
            days of observations by 14 understates the baseline by ~4.7x and
            inflates every buzz score by the same factor.

    Clamped to [1.0, BASELINE_DAYS]. The lower clamp stops a sub-one-day store
    from multiplying the rate; the upper stops a coverage figure larger than the
    window from diluting it.
    """
    days = min(max(coverage_days, 1.0), float(BASELINE_DAYS))
    return round(baseline_count / days, 3)


def baseline_coverage_days(session: Session, now: datetime | None = None) -> float:
    """Days of history observed *before* the trailing window.

    This is what tells us whether a baseline means anything. It measures from the
    earliest record we hold (clamped to the baseline window start) up to where the
    trailing window begins — so a store that has only ever ingested once returns
    0.0, and buzz scoring is correctly refused rather than fabricated.
    """
    baseline_start, window_start, now = window_bounds(now)

    earliest = session.execute(select(func.min(Mention.created_utc))).scalar_one_or_none()
    if earliest is None:
        return 0.0
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)

    effective_start = max(earliest, baseline_start)
    if effective_start >= window_start:
        return 0.0
    return round((window_start - effective_start).total_seconds() / 86400, 2)


def is_warming_up(coverage_days: float) -> bool:
    """Whether the store has too little history for buzz scores to mean anything."""
    return coverage_days < MIN_BASELINE_COVERAGE_DAYS


def is_provisional(coverage_days: float) -> bool:
    """Whether the baseline is partial enough to be worth warning about.

    Tolerance of PROVISIONAL_TOLERANCE_DAYS rather than a bare
    `coverage < BASELINE_DAYS`. A store holding 13.96 days of history is not
    meaningfully provisional, but a strict comparison flags it — and then the
    message rounds to "only 14.0 of 14 days observed", which reads as a bug and
    trains the reader to ignore the warning. A warning that fires when nothing is
    wrong is worse than no warning: it costs the real one its credibility.

    The score itself is unaffected either way — compute_baseline_avg already
    divides by actual observed days. This governs only whether we say something.
    """
    return not is_warming_up(coverage_days) and coverage_days < (
        BASELINE_DAYS - PROVISIONAL_TOLERANCE_DAYS
    )


def warmup_note(coverage_days: float) -> str:
    """Explain the warm-up state in terms a caller can act on."""
    return (
        f"Not enough history to detect buzz yet. Buzz is a comparison against a ticker's "
        f"own normal, and only {coverage_days:.1f} days of baseline history have been "
        f"observed (need {MIN_BASELINE_COVERAGE_DAYS:.0f}, ideally {BASELINE_DAYS}). "
        "Reporting scores now would mean dividing by a baseline of roughly zero and "
        "calling every widely-discussed ticker a spike. Leave the worker running — "
        "mention counts via buzz_get_ticker are already accurate and usable."
    )


def provisional_note(coverage_days: float) -> str:
    """Flag scores computed against a partial baseline."""
    return (
        f"PROVISIONAL: only {coverage_days:.1f} of {BASELINE_DAYS} baseline days have been "
        "observed, so scores are computed against actual observed history. They will "
        "shift as the baseline fills in. Treat the ranking as directional."
    )


def criteria_text() -> str:
    """Human-readable statement of the thresholds, echoed in tool output.

    Surfacing the criteria alongside the results means Claude can explain *why*
    a ticker is on the list without the caller having to read the source.
    """
    return (
        f"Flagged when BOTH: 24h mentions >= {BUZZ_MIN_MENTIONS_24H} "
        f"AND buzz_score >= {BUZZ_SCORE_THRESHOLD}x the {BASELINE_DAYS}-day trailing daily average "
        f"(baseline window excludes the trailing {TRAILING_HOURS}h)."
    )


# ---------------------------------------------------------------------------
# Database-backed aggregation
# ---------------------------------------------------------------------------


def _sentiment_counts(
    session: Session, ticker: str, start: datetime, end: datetime
) -> tuple[int, int, int, int]:
    """Return (bullish, bearish, neutral, unclassified) for a ticker in a window."""
    rows = session.execute(
        select(Mention.sentiment, func.count(Mention.id))
        .where(
            Mention.ticker == ticker,
            Mention.created_utc >= start,
            Mention.created_utc <= end,
        )
        .group_by(Mention.sentiment)
    ).all()
    counts = {label: n for label, n in rows}
    return (
        counts.get("bullish", 0),
        counts.get("bearish", 0),
        counts.get("neutral", 0),
        counts.get(None, 0),
    )


def _sample_urls(
    session: Session, ticker: str, start: datetime, end: datetime, limit: int = 3
) -> list[str]:
    """Representative post URLs, highest-engagement first.

    Engagement is nullable (not every source exposes a score), so nulls sort last
    and recency breaks the tie. The goal is a link a human can click to sanity-check
    the number — so the *most visible* posts are the right ones to surface.
    """
    rows = session.execute(
        select(Mention.url)
        .where(
            Mention.ticker == ticker,
            Mention.created_utc >= start,
            Mention.created_utc <= end,
        )
        .order_by(
            Mention.engagement.is_(None),  # False (0) sorts before True (1): non-null first
            Mention.engagement.desc(),
            Mention.created_utc.desc(),
        )
        .limit(limit)
    ).all()
    return [r[0] for r in rows]


def _source_breakdown(
    session: Session, ticker: str, start: datetime, end: datetime
) -> list[SourceBreakdown]:
    """Per-source counts for a ticker in a window, most mentions first."""
    rows = session.execute(
        select(Mention.source, func.count(Mention.id))
        .where(
            Mention.ticker == ticker,
            Mention.created_utc >= start,
            Mention.created_utc <= end,
        )
        .group_by(Mention.source)
        .order_by(func.count(Mention.id).desc())
    ).all()
    out: list[SourceBreakdown] = []
    for source, n in rows:
        try:
            out.append(SourceBreakdown(source=Source(source), mention_count=n))
        except ValueError:
            # Unknown source string in the DB (schema drift) — skip rather than 500.
            continue
    return out


def _has_synthetic(session: Session, ticker: str, start: datetime, end: datetime) -> bool:
    """Whether any mention backing this ticker's window is seeded demo data."""
    return bool(
        session.execute(
            select(func.count(Mention.id)).where(
                Mention.ticker == ticker,
                Mention.created_utc >= start,
                Mention.created_utc <= end,
                Mention.is_synthetic.is_(True),
            )
        ).scalar_one()
    )


def trend_7d(session: Session, ticker: str, now: datetime | None = None) -> list[TrendPoint]:
    """Daily mention counts for the last 7 UTC days, oldest first.

    Days with no mentions are zero-filled rather than omitted — a caller plotting
    this should see the gap, not a compressed line that implies continuity.
    """
    now = now or utcnow()
    start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)

    rows = session.execute(
        select(Mention.created_utc).where(
            Mention.ticker == ticker, Mention.created_utc >= start, Mention.created_utc <= now
        )
    ).all()

    buckets: dict[str, int] = {}
    for (created,) in rows:
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        key = created.astimezone(timezone.utc).date().isoformat()
        buckets[key] = buckets.get(key, 0) + 1

    points: list[TrendPoint] = []
    for offset in range(7):
        day = (start + timedelta(days=offset)).date().isoformat()
        points.append(TrendPoint(date=day, mention_count=buckets.get(day, 0)))
    return points


def ticker_stats(session: Session, ticker: str, now: datetime | None = None) -> dict:
    """Full stat block for one ticker. Backs buzz_get_ticker.

    Unlike list_trending, this still returns during warm-up. Mention counts, the
    source breakdown, the 7-day trend, and sentiment are all accurate from the
    first tick — only the *comparison to normal* needs history. So the counts are
    reported and `is_buzzing` is forced False with an explanatory note, rather
    than withholding data that is genuinely correct.
    """
    baseline_start, window_start, now = window_bounds(now)

    mentions_24h = session.execute(
        select(func.count(Mention.id)).where(
            Mention.ticker == ticker,
            Mention.created_utc >= window_start,
            Mention.created_utc <= now,
        )
    ).scalar_one()

    baseline_count = session.execute(
        select(func.count(Mention.id)).where(
            Mention.ticker == ticker,
            Mention.created_utc >= baseline_start,
            Mention.created_utc < window_start,
        )
    ).scalar_one()

    coverage = baseline_coverage_days(session, now)
    warming = is_warming_up(coverage)

    baseline_avg = compute_baseline_avg(baseline_count, coverage)
    score = 0.0 if warming else compute_buzz_score(mentions_24h, baseline_avg)
    bullish, bearish, neutral, unclassified = _sentiment_counts(session, ticker, window_start, now)

    notes: list[str] = []
    if warming:
        notes.append(warmup_note(coverage))
    elif is_provisional(coverage):
        notes.append(provisional_note(coverage))

    return {
        "ticker": ticker,
        "is_buzzing": False if warming else is_buzzing(mentions_24h, score),
        "mention_count_24h": mentions_24h,
        "buzz_score": score,
        "baseline_daily_avg": baseline_avg,
        "sentiment": aggregate_sentiment(bullish, bearish, neutral, unclassified),
        "source_breakdown": _source_breakdown(session, ticker, window_start, now),
        "trend_7d": trend_7d(session, ticker, now),
        "sample_post_urls": _sample_urls(session, ticker, window_start, now),
        "is_synthetic": _has_synthetic(session, ticker, window_start, now),
        "notes": notes,
    }


def list_trending(session: Session, limit: int = 20, now: datetime | None = None) -> list[dict]:
    """Every currently-buzzing ticker, highest buzz_score first. Backs buzz_list_trending.

    Returns an EMPTY list while the store is warming up. There is no honest
    trending list without a baseline — on a cold store every ticker divides by
    the floor, and the tool confidently reports AAPL at 96x normal on an ordinary
    Tuesday. Callers should pair this with `baseline_coverage_days()` and surface
    `warmup_note()`.

    Only tickers clearing the absolute-mentions floor are scored at all — applied
    in SQL (HAVING) so the expensive per-ticker work runs on a handful of
    candidates rather than the whole universe.
    """
    baseline_start, window_start, now = window_bounds(now)

    coverage = baseline_coverage_days(session, now)
    if is_warming_up(coverage):
        return []

    candidates = session.execute(
        select(Mention.ticker, func.count(Mention.id).label("n"))
        .where(Mention.created_utc >= window_start, Mention.created_utc <= now)
        .group_by(Mention.ticker)
        .having(func.count(Mention.id) >= BUZZ_MIN_MENTIONS_24H)
    ).all()

    results: list[dict] = []
    for ticker, mentions_24h in candidates:
        baseline_count = session.execute(
            select(func.count(Mention.id)).where(
                Mention.ticker == ticker,
                Mention.created_utc >= baseline_start,
                Mention.created_utc < window_start,
            )
        ).scalar_one()
        baseline_avg = compute_baseline_avg(baseline_count, coverage)
        score = compute_buzz_score(mentions_24h, baseline_avg)

        if not is_buzzing(mentions_24h, score):
            continue

        bullish, bearish, neutral, unclassified = _sentiment_counts(
            session, ticker, window_start, now
        )
        breakdown = _source_breakdown(session, ticker, window_start, now)

        results.append(
            {
                "ticker": ticker,
                "mention_count_24h": mentions_24h,
                "buzz_score": score,
                "baseline_daily_avg": baseline_avg,
                "sentiment": aggregate_sentiment(bullish, bearish, neutral, unclassified),
                "top_source": breakdown[0].source if breakdown else Source.REDDIT,
                "sample_post_urls": _sample_urls(session, ticker, window_start, now),
                "is_synthetic": _has_synthetic(session, ticker, window_start, now),
            }
        )

    results.sort(key=lambda r: r["buzz_score"], reverse=True)
    return results[:limit]
