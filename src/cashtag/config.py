"""Central configuration. Every tunable in Cashtag lives here.

Buzz thresholds are module-level constants (not env vars) on purpose: they are
part of the *methodology*, and a methodology that silently differs between
environments is not a methodology. Change them here, in a commit, with a reason.
Operational knobs (credentials, budget, poll cadence) are environment-driven.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Buzz methodology
# ---------------------------------------------------------------------------

#: A ticker is "buzzing" only if BOTH thresholds are met.
#: Ratio alone promotes noise (1 -> 3 mentions/day is a 3.0x score but not a signal),
#: absolute alone just ranks mega-caps that are always talked about.
BUZZ_SCORE_THRESHOLD = 2.0
BUZZ_MIN_MENTIONS_24H = 15

#: Trailing window used to compute the "normal" daily rate for a ticker.
BASELINE_DAYS = 14

#: The baseline window ENDS where the 24h window BEGINS — the two never overlap.
#: If the spike were included in its own baseline it would inflate the denominator
#: and suppress the very score we are trying to detect. See README "The buzz methodology".
TRAILING_HOURS = 24

#: Floor on the baseline denominator, in mentions/day. Without it, a ticker with
#: no history divides by zero and scores infinity — every never-before-seen
#: ticker would top the list forever. 0.5 means a cold-start ticker's score is
#: capped at 2x its raw 24h count, which keeps it rankable but not dominant.
BASELINE_FLOOR = 0.5

#: Minimum days of observed history BEFORE the trailing window required to score
#: anything at all.
#:
#: This exists because of a real failure found on first live run. On a fresh
#: store every ticker has a baseline of 0, which floors to BASELINE_FLOOR and
#: makes buzz_score = 2x the raw count. The result: 18 tickers "buzzing", with
#: AAPL — the most-discussed stock in the world, on an ordinary day — reported at
#: 96x normal. Every number was garbage, and none of it looked like an error.
#:
#: Buzz is a comparison against normal. With no history there is no "normal", so
#: the honest output is "not enough history yet", not a confident wrong answer.
#: Below this threshold, buzz_list_trending returns nothing and explains why.
#: Between this and BASELINE_DAYS, scores are computed against ACTUAL observed
#: days (not an assumed 14) and marked provisional.
MIN_BASELINE_COVERAGE_DAYS = 3.0

#: Slack before a baseline is called "provisional" in tool output. Real ingestion
#: never lands on an exact day boundary, so a strict `coverage < BASELINE_DAYS`
#: fires at 13.96/14 days and prints "only 14.0 of 14 days observed" — a warning
#: that reads as a bug and teaches the reader to ignore warnings. Affects the
#: message only; the score always divides by actual observed days.
PROVISIONAL_TOLERANCE_DAYS = 1.0

# ---------------------------------------------------------------------------
# Ingestion cadence
# ---------------------------------------------------------------------------

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)
POLL_INTERVAL_MARKET_HOURS_MIN = 15
POLL_INTERVAL_OFF_HOURS_MIN = 60

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: Haiku 4.5: cheap and fast, correctly sized for high-volume 3-way classification.
#: Using the alias rather than the dated ID (claude-haiku-4-5-20251001) — same
#: model, one less string to update.
CLASSIFIER_MODEL = "claude-haiku-4-5"

#: Posts packed into a single Haiku request. Batching here (rather than via the
#: Batches API) keeps labels fresh within one poll cycle while still amortizing
#: the prompt across many posts. The Batches API would halve cost but can take
#: up to an hour, which does not fit a 15-minute ingest loop.
CLASSIFY_BATCH_SIZE = 20

#: Max posts labelled per worker tick. Bounds spend if a backlog builds up.
CLASSIFY_MAX_PER_RUN = 400

#: Post text is truncated before classification. Sentiment lives in the first
#: paragraph; DD posts run to thousands of words and we pay for every token.
CLASSIFY_TEXT_CHAR_LIMIT = 1200

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "StockMarket",
]

#: Reddit free non-commercial tier: 100 requests/min. We stay well under.
REDDIT_POSTS_PER_SUBREDDIT = 100
REDDIT_RATE_LIMIT_PER_MIN = 100

#: X pay-per-use pricing as of Feb 2026. Used for the budget ledger.
X_COST_PER_READ_USD = 0.005
X_MAX_RESULTS_PER_QUERY = 100

#: Bright Data: lowest-confidence, lowest-volume sources. See README "Four sources, four access models".
BRIGHTDATA_POST_LIMIT = 50


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    database_url: str = field(
        default_factory=lambda: os.getenv("CASHTAG_DATABASE_URL", "sqlite:///cashtag.db")
    )
    auth_token: str | None = field(default_factory=lambda: os.getenv("CASHTAG_AUTH_TOKEN") or None)
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )

    reddit_client_id: str | None = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_ID") or None
    )
    reddit_client_secret: str | None = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET") or None
    )
    reddit_user_agent: str = field(
        default_factory=lambda: os.getenv("REDDIT_USER_AGENT", "python:cashtag:0.1.0")
    )
    subreddits: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("CASHTAG_SUBREDDITS", ",".join(DEFAULT_SUBREDDITS)).split(",")
            if s.strip()
        ]
    )

    x_enabled: bool = field(default_factory=lambda: _env_bool("X_ENABLED", False))
    x_bearer_token: str | None = field(default_factory=lambda: os.getenv("X_BEARER_TOKEN") or None)
    x_monthly_budget_usd: float = field(
        default_factory=lambda: _env_float("X_MONTHLY_BUDGET_USD", 25.0)
    )

    brightdata_enabled: bool = field(default_factory=lambda: _env_bool("BRIGHTDATA_ENABLED", False))
    brightdata_api_key: str | None = field(
        default_factory=lambda: os.getenv("BRIGHTDATA_API_KEY") or None
    )

    stocktwits_enabled: bool = field(default_factory=lambda: _env_bool("STOCKTWITS_ENABLED", True))

    @property
    def reddit_configured(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)

    @property
    def classifier_configured(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
