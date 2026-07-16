"""Pydantic models for MCP tool inputs and outputs.

A note on the percentage fields, because they are easy to misread
--------------------------------------------------------------------
`bullish_pct` and `bearish_pct` are a *directional split*: they are computed
over (bullish + bearish) only and sum to 100 by construction. 100 means every
opinionated post was bullish; 0 means every one was bearish; 50 means evenly
split. Neutral posts are deliberately excluded from that denominator so that a
flood of neutral chatter cannot drag a strongly bullish ticker toward the middle.

`neutral_pct` is computed over a DIFFERENT denominator — all classified posts —
and answers a separate question: "how much of this conversation is actually an
opinion at all?" It is a data-quality signal, not a third slice of the
directional split.

The consequence, stated plainly because it will otherwise look like a bug:
**the three percentages do not sum to 100.** That is intended. If you want a
three-way share that does sum to 100, divide the raw counts
(`bullish_count`, `bearish_count`, `neutral_count`) yourself.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Sentiment(str, Enum):
    """Label assigned to a single post by the classifier."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Source(str, Enum):
    """Where a mention came from. Each uses a different access method by necessity."""

    REDDIT = "reddit"
    X = "x"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    STOCKTWITS = "stocktwits"


# ---------------------------------------------------------------------------
# Tool inputs
# ---------------------------------------------------------------------------


class ListTrendingInput(BaseModel):
    """Input for buzz_list_trending."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    limit: int = Field(
        default=20,
        description="Maximum number of buzzing tickers to return, 1-100 (e.g. 20).",
        ge=1,
        le=100,
    )


class GetTickerInput(BaseModel):
    """Input for buzz_get_ticker."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    ticker: str = Field(
        ...,
        description="Ticker symbol, with or without a leading '$' (e.g. 'TSLA', '$GME', 'nvda').",
        min_length=1,
        max_length=10,
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        cleaned = v.strip().lstrip("$").upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty or just '$'")
        if not cleaned.isalpha():
            raise ValueError(
                f"Ticker must be letters only, got {v!r}. "
                "Use the plain symbol, e.g. 'TSLA' — not an option contract or a company name."
            )
        return cleaned


# ---------------------------------------------------------------------------
# Tool outputs
# ---------------------------------------------------------------------------


class SentimentSplit(BaseModel):
    """Sentiment breakdown for a ticker. Read the module docstring on denominators."""

    bullish_pct: float | None = Field(
        None,
        description=(
            "Share of OPINIONATED posts that were bullish: "
            "bullish / (bullish + bearish) * 100. 100 = fully bullish, 0 = fully bearish. "
            "Null when there are no opinionated posts (all neutral or none classified)."
        ),
    )
    bearish_pct: float | None = Field(
        None,
        description="100 - bullish_pct. Null under the same condition as bullish_pct.",
    )
    neutral_pct: float | None = Field(
        None,
        description=(
            "Share of ALL classified posts that were neutral: neutral / total * 100. "
            "Different denominator from bullish_pct/bearish_pct — this does NOT sum with them "
            "to 100. High values mean the chatter is mostly non-directional."
        ),
    )
    bullish_count: int = Field(0, description="Raw count of bullish-labelled posts in the window.")
    bearish_count: int = Field(0, description="Raw count of bearish-labelled posts in the window.")
    neutral_count: int = Field(0, description="Raw count of neutral-labelled posts in the window.")
    unclassified_count: int = Field(
        0,
        description=(
            "Posts ingested but not yet labelled. If this is large relative to the counts above, "
            "the sentiment split is based on a partial sample — treat it with caution."
        ),
    )


class SourceBreakdown(BaseModel):
    """Per-source mention counts."""

    source: Source = Field(..., description="The platform the mentions came from.")
    mention_count: int = Field(..., description="Mentions from this source in the window.")


class TrendPoint(BaseModel):
    """One day of the mention trend."""

    date: str = Field(..., description="Calendar date in UTC, ISO format (e.g. '2026-07-15').")
    mention_count: int = Field(..., description="Mentions recorded on that date.")


class TrendingTicker(BaseModel):
    """A ticker currently flagged as buzzing."""

    ticker: str = Field(..., description="Ticker symbol, uppercase, no '$' (e.g. 'TSLA').")
    mention_count_24h: int = Field(..., description="Mentions in the trailing 24 hours.")
    buzz_score: float = Field(
        ...,
        description=(
            "mention_count_24h divided by the 14-day trailing daily average. "
            "2.0 means twice the normal chatter for this ticker. The baseline window "
            "excludes the trailing 24h, so a spike does not dilute its own score."
        ),
    )
    baseline_daily_avg: float = Field(
        ...,
        description="Mean mentions/day over the 14 days preceding the trailing-24h window.",
    )
    sentiment: SentimentSplit = Field(..., description="Sentiment breakdown over the same 24h.")
    top_source: Source = Field(..., description="Source contributing the most mentions in 24h.")
    sample_post_urls: list[str] = Field(
        default_factory=list,
        description="2-3 representative post URLs, highest-engagement first, for spot-checking.",
    )
    is_synthetic: bool = Field(
        False,
        description=(
            "True when this row is backed by seeded demo data rather than live ingestion. "
            "Never present in production; used so the connector is demoable before "
            "source credentials are approved."
        ),
    )


class TrendingResponse(BaseModel):
    """Output of buzz_list_trending."""

    count: int = Field(..., description="Number of tickers returned.")
    generated_at: str = Field(..., description="UTC ISO timestamp of this snapshot.")
    criteria: str = Field(..., description="Human-readable statement of the flagging thresholds.")
    tickers: list[TrendingTicker] = Field(
        default_factory=list, description="Buzzing tickers, highest buzz_score first."
    )
    data_freshness: str = Field(..., description="How recently ingestion last wrote a record.")
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats worth surfacing to the user (synthetic data, stale ingest, etc.).",
    )


class TickerDetail(BaseModel):
    """Output of buzz_get_ticker. Returned for any ticker, buzzing or not."""

    ticker: str = Field(..., description="Ticker symbol, uppercase, no '$'.")
    is_buzzing: bool = Field(
        ..., description="Whether this ticker currently meets BOTH flagging thresholds."
    )
    mention_count_24h: int = Field(..., description="Mentions in the trailing 24 hours.")
    buzz_score: float = Field(..., description="24h mentions / 14-day trailing daily average.")
    baseline_daily_avg: float = Field(
        ..., description="Mean mentions/day over the baseline window."
    )
    sentiment: SentimentSplit = Field(..., description="Sentiment breakdown over the same 24h.")
    source_breakdown: list[SourceBreakdown] = Field(
        default_factory=list, description="Where the 24h mentions came from, most first."
    )
    trend_7d: list[TrendPoint] = Field(
        default_factory=list,
        description="Daily mention counts for the last 7 UTC days, oldest first. Gaps are zero-filled.",
    )
    sample_post_urls: list[str] = Field(
        default_factory=list, description="2-3 representative post URLs."
    )
    is_synthetic: bool = Field(False, description="True when backed by seeded demo data.")
    notes: list[str] = Field(default_factory=list, description="Caveats worth surfacing.")
