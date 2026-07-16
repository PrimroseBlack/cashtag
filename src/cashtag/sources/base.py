"""Shared contract for ingestion sources.

The abstraction is deliberately thin. Each source has a genuinely different
access model — OAuth SDK, metered REST, scraping vendor, open endpoint — and
flattening them behind a fat base class would hide the exact differences that
matter operationally (who rate-limits, who charges, who can be trusted).

So the contract is only: "produce RawPosts". Everything else — auth, pagination,
budget, retries — belongs to the source that has that problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from ..models import Source


@dataclass(frozen=True)
class RawPost:
    """A single post from any source, before ticker extraction.

    Attributes:
        source: Which platform.
        source_id: Native post ID. Must be stable — it is the dedupe key.
        text: Title + body, or caption. What gets ticker-scanned and classified.
        url: Permalink for humans to click.
        created_utc: When the POST was authored (tz-aware). Not when we fetched it.
        author: Username, if exposed.
        subsource: Sub-feed (subreddit, hashtag), if meaningful.
        engagement: Upvotes/likes, normalized. None when unavailable.
    """

    source: Source
    source_id: str
    text: str
    url: str
    created_utc: datetime
    author: str | None = None
    subsource: str | None = None
    engagement: int | None = None


class IngestionSource(ABC):
    """Base class for a pollable source."""

    #: Set False when credentials or a feature flag are missing. The worker skips
    #: disabled sources without treating it as an error — a connector missing one
    #: of five sources should degrade, not fail.
    enabled: bool = True

    @property
    @abstractmethod
    def name(self) -> Source:
        """Which source this adapter fetches."""

    @abstractmethod
    def fetch(self) -> list[RawPost]:
        """Fetch recent posts.

        Implementations must not raise on ordinary upstream failure — log and
        return what was collected. One source being down is not a reason to lose
        the other four sources' data for that tick.
        """

    def status(self) -> str:
        """One-line health string for logs and the health endpoint."""
        return "enabled" if self.enabled else "disabled (missing credentials or flag)"
