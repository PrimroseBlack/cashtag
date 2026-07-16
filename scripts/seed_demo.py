#!/usr/bin/env python3
"""Seed synthetic mention data so the connector is demoable before credentials land.

Why this exists
---------------
Reddit's free-tier approval takes 2-4 weeks. Without seed data the connector is
un-demoable until then, which is a bad place to be when the artifact's whole
purpose is being shown to people. This generates a realistic store so
`buzz_list_trending` returns something the moment you install it.

Every row is written with `is_synthetic=True`, and every tool response that
touches a synthetic row carries a loud SYNTHETIC DATA warning. That is not
decoration. Fake market signal that looks real is genuinely dangerous — someone
could act on it. The flag makes that structurally impossible to miss.

The dataset is designed to exercise the methodology, not just fill a table:

  GME   score ~12   buzzing      — the obvious spike
  SOFI  score ~14   buzzing      — small-cap spike from a quiet baseline
  PLTR  score ~6    buzzing      — mid-size spike, mixed sentiment
  AMC   score ~5    buzzing      — spike with bearish lean
  NVDA  score ~2.3  buzzing      — just over the ratio line, high volume
  TSLA  score ~1.1  NOT buzzing  — high volume, no anomaly (absolute alone would flag this)
  AAPL  score ~1.1  NOT buzzing  — same
  RIVN  score ~12   NOT buzzing  — great ratio, only 12 mentions (ratio alone would flag this)

TSLA and RIVN are the point. They are the two tickers that a single-threshold
system gets wrong, and they are in the demo so you can show that on a call.

Usage:
    python scripts/seed_demo.py            # seed (refuses if real data present)
    python scripts/seed_demo.py --reset    # wipe and reseed
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import func, select  # noqa: E402

from cashtag.db import Mention, init_db, reset_db, session_scope, utcnow  # noqa: E402
from cashtag.models import Sentiment, Source  # noqa: E402

# Fixed seed: the demo must show the same numbers every time it is run.
RNG = random.Random(20260715)

# ticker -> (baseline_per_day, mentions_24h, bullish_weight, bearish_weight, neutral_weight)
PROFILES = {
    "GME": (8, 95, 0.62, 0.22, 0.16),
    "SOFI": (3, 42, 0.71, 0.14, 0.15),
    "PLTR": (5, 31, 0.48, 0.37, 0.15),
    "AMC": (4, 22, 0.31, 0.54, 0.15),
    "NVDA": (30, 68, 0.66, 0.20, 0.14),
    "TSLA": (40, 45, 0.44, 0.42, 0.14),
    "AAPL": (25, 28, 0.52, 0.30, 0.18),
    "RIVN": (1, 12, 0.58, 0.26, 0.16),
}

SOURCE_MIX = [
    (Source.REDDIT, 0.58),
    (Source.X, 0.24),
    (Source.STOCKTWITS, 0.12),
    (Source.INSTAGRAM, 0.04),
    (Source.TIKTOK, 0.02),
]

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]

BULLISH_TEMPLATES = [
    "${t} is setting up beautifully here. Loaded up on calls this morning.",
    "Just added to my ${t} position. This dip is a gift.",
    "${t} to the moon. Diamond hands, not selling a single share.",
    "The ${t} thesis is intact. Buying more every paycheck.",
    "${t} breaking out of the range on volume. This is the one.",
    "Bought ${t} leaps today. Feeling good about the next 6 months.",
    "${t} earnings are going to blow out estimates. Calls printing.",
    "Why is nobody talking about ${t}? Massively undervalued here.",
]

BEARISH_TEMPLATES = [
    "${t} is a bubble and everyone knows it. Puts loaded.",
    "Sold my entire ${t} position today. The fundamentals don't support this.",
    "${t} is going to get destroyed at earnings. Short it.",
    "Anyone else think ${t} is way overextended? Taking profits.",
    "${t} chart looks like a textbook top. I'm out.",
    "Bought puts on ${t}. This runup makes no sense.",
    "${t} insiders are dumping. That tells you everything.",
]

NEUTRAL_TEMPLATES = [
    "What's everyone's take on ${t}? Thinking about starting a position.",
    "${t} announced a new partnership today. Thoughts?",
    "Can someone explain the ${t} situation to me? New here.",
    "${t} earnings call is tomorrow. Anyone listening in?",
    "Here's my portfolio, ${t} is about 8% of it. Rate it.",
    "Article about ${t} from this morning, sharing for discussion.",
]


def _pick_source() -> Source:
    roll = RNG.random()
    cumulative = 0.0
    for source, weight in SOURCE_MIX:
        cumulative += weight
        if roll <= cumulative:
            return source
    return Source.REDDIT


def _pick_sentiment(bull_w: float, bear_w: float) -> Sentiment:
    roll = RNG.random()
    if roll < bull_w:
        return Sentiment.BULLISH
    if roll < bull_w + bear_w:
        return Sentiment.BEARISH
    return Sentiment.NEUTRAL


def _make_text(ticker: str, sentiment: Sentiment) -> str:
    pool = {
        Sentiment.BULLISH: BULLISH_TEMPLATES,
        Sentiment.BEARISH: BEARISH_TEMPLATES,
        Sentiment.NEUTRAL: NEUTRAL_TEMPLATES,
    }[sentiment]
    return RNG.choice(pool).replace("${t}", f"${ticker}")


def _make_mention(ticker: str, sentiment: Sentiment, created, counter: int) -> Mention:
    source = _pick_source()
    text = _make_text(ticker, sentiment)
    source_id = f"synthetic-{ticker}-{counter}"

    urls = {
        Source.REDDIT: f"https://reddit.com/r/wallstreetbets/comments/{source_id}",
        Source.X: f"https://x.com/i/web/status/{abs(hash(source_id)) % 10**18}",
        Source.STOCKTWITS: f"https://stocktwits.com/message/{abs(hash(source_id)) % 10**9}",
        Source.INSTAGRAM: f"https://instagram.com/p/{source_id}",
        Source.TIKTOK: f"https://tiktok.com/@demo/video/{abs(hash(source_id)) % 10**18}",
    }

    # StockTwits is the only source with author self-tags, and only ~35% of its
    # posts carry one. Mirroring that here keeps the eval script's sample size
    # realistic rather than flattering.
    #
    # The tag disagrees with the label ~12% of the time ON PURPOSE. Setting
    # author_sentiment == sentiment would make eval_classifier.py report a
    # perfect 100%, which measures nothing and is actively misleading — it is a
    # tautology dressed as a metric. Injecting realistic disagreement means the
    # eval script's output *looks* like what real data produces, so the plumbing
    # is exercised rather than short-circuited. The resulting number is still
    # meaningless (it is noise we invented); eval_classifier.py says so when it
    # detects synthetic rows.
    author_sentiment = None
    if source == Source.STOCKTWITS and sentiment != Sentiment.NEUTRAL and RNG.random() < 0.35:
        if RNG.random() < 0.12:
            author_sentiment = (
                Sentiment.BEARISH.value
                if sentiment == Sentiment.BULLISH
                else Sentiment.BULLISH.value
            )
        else:
            author_sentiment = sentiment.value

    return Mention(
        ticker=ticker,
        source=source.value,
        source_id=source_id,
        subsource=RNG.choice(SUBREDDITS) if source == Source.REDDIT else None,
        author=f"demo_user_{RNG.randint(1000, 9999)}",
        text=text,
        url=urls[source],
        engagement=RNG.randint(0, 4200),
        created_utc=created,
        sentiment=sentiment.value,
        sentiment_model="synthetic-seed",
        classified_at=created,
        author_sentiment=author_sentiment,
        is_synthetic=True,
    )


def seed() -> int:
    """Generate the synthetic store. Returns rows written."""
    now = utcnow()
    written = 0

    with session_scope() as session:
        counter = 0
        for ticker, (baseline, spike, bull_w, bear_w, _) in PROFILES.items():
            # Baseline: 14 days ending where the trailing-24h window begins.
            # Must match buzz.window_bounds() or the demo's scores won't reproduce.
            for day in range(1, 15):
                for _ in range(baseline):
                    counter += 1
                    created = now - timedelta(
                        days=day, hours=RNG.uniform(0, 23), minutes=RNG.uniform(0, 59)
                    )
                    sentiment = _pick_sentiment(bull_w, bear_w)
                    session.add(_make_mention(ticker, sentiment, created, counter))
                    written += 1

            # The trailing-24h window itself.
            for _ in range(spike):
                counter += 1
                created = now - timedelta(hours=RNG.uniform(0, 23.5))
                sentiment = _pick_sentiment(bull_w, bear_w)
                session.add(_make_mention(ticker, sentiment, created, counter))
                written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic demo data for Cashtag.")
    parser.add_argument(
        "--reset", action="store_true", help="Drop and recreate all tables before seeding."
    )
    args = parser.parse_args()

    if args.reset:
        reset_db()
        print("Dropped and recreated all tables.")
    else:
        init_db()
        with session_scope() as session:
            real = session.execute(
                select(func.count(Mention.id)).where(Mention.is_synthetic.is_(False))
            ).scalar_one()
        if real:
            # Never silently mix fake rows into a store holding real observations.
            print(
                f"Refusing to seed: the store already holds {real} REAL mention rows.\n"
                "Seeding would mix synthetic data into live data. Use --reset to wipe "
                "everything first, or point CASHTAG_DATABASE_URL at a scratch database.",
                file=sys.stderr,
            )
            sys.exit(1)

    count = seed()
    print(f"Seeded {count} synthetic mention rows across {len(PROFILES)} tickers.\n")

    from cashtag.buzz import list_trending

    with session_scope() as session:
        trending = list_trending(session, limit=20)

    print(f"{'TICKER':<8}{'24H':>6}{'BASE/D':>9}{'SCORE':>8}{'BULL%':>8}  STATUS")
    print("-" * 56)
    flagged = {row["ticker"] for row in trending}
    for row in trending:
        bull = row["sentiment"].bullish_pct
        print(
            f"{row['ticker']:<8}{row['mention_count_24h']:>6}"
            f"{row['baseline_daily_avg']:>9.2f}{row['buzz_score']:>8.2f}"
            f"{(f'{bull:.0f}' if bull is not None else 'n/a'):>8}  BUZZING"
        )
    for ticker in PROFILES:
        if ticker not in flagged:
            print(f"{ticker:<8}{'':>6}{'':>9}{'':>8}{'':>8}  not flagged")

    print(
        "\nAll rows are marked is_synthetic=True and every tool response that "
        "returns one will say so."
    )


if __name__ == "__main__":
    main()
