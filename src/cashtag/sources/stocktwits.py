"""StockTwits ingestion — free, no auth, and the only labelled data in the system.

Access model: open REST endpoint, no key, no OAuth. The cheapest source to add,
which is why it was the stretch goal.

But it earns its place for a reason beyond cost. StockTwits posts are
**self-tagged bullish/bearish by their authors**, which makes this the only
source that arrives with ground truth attached. That gives the project something
most sentiment demos conspicuously lack: a way to know whether the classifier is
any good.

So the author's tag is stored in `author_sentiment` and deliberately NOT copied
into `sentiment`. The classifier labels these posts blind, exactly as it labels
everything else, and `scripts/eval_classifier.py` then measures agreement between
the two columns. Shortcutting — trusting the self-tag as the label — would save a
few cents of Haiku calls and destroy the only evaluation set available.

Caveats worth knowing before over-trusting the ground truth:
  - Tagging is optional; most posts are untagged, and those get `author_sentiment
    = None` (they still get classified, they just aren't scoreable).
  - StockTwits has no "neutral" tag — only Bullish and Bearish. So agreement can
    only be measured on the directional call, never on neutral recall.
  - Self-tags are self-reported by people talking their own book. This is a
    reasonable proxy for intent, not an objective label.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from ..config import settings
from ..models import Sentiment, Source
from .base import IngestionSource, RawPost

logger = logging.getLogger(__name__)

STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

#: Unauthenticated StockTwits is rate-limited (~200 req/hr). Poll a focused set of
#: symbols per tick rather than the whole universe.
MAX_SYMBOLS_PER_RUN = 30
REQUEST_TIMEOUT = 15.0


class StockTwitsSource(IngestionSource):
    """Polls per-symbol streams. Returns (post, author_sentiment) pairs."""

    def __init__(self, tickers: list[str], client=None):
        self.tickers = tickers[:MAX_SYMBOLS_PER_RUN]
        self._client = client
        self.enabled = settings.stocktwits_enabled

    @property
    def name(self) -> Source:
        return Source.STOCKTWITS

    def fetch(self) -> list[RawPost]:
        """Fetch posts only. Use `fetch_with_tags` to keep the self-tags."""
        return [post for post, _ in self.fetch_with_tags()]

    def fetch_with_tags(self) -> list[tuple[RawPost, Sentiment | None]]:
        """Fetch posts alongside each author's self-declared tag.

        Returns:
            (RawPost, author_sentiment) pairs. author_sentiment is None for the
            majority of posts — tagging is optional on StockTwits.
        """
        if not self.enabled:
            return []

        client = self._client or httpx.Client(timeout=REQUEST_TIMEOUT)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=26)
        results: list[tuple[RawPost, Sentiment | None]] = []

        for symbol in self.tickers:
            try:
                response = client.get(STREAM_URL.format(symbol=symbol))
                if response.status_code == 404:
                    # Symbol not covered by StockTwits. Normal, not an error.
                    continue
                if response.status_code == 429:
                    logger.warning(
                        "StockTwits rate limit hit; stopping after %d symbols", len(results)
                    )
                    break
                response.raise_for_status()
                messages = response.json().get("messages", []) or []
            except Exception as exc:
                logger.warning("StockTwits fetch failed for %s: %s", symbol, exc)
                continue

            for msg in messages:
                created_raw = msg.get("created_at")
                if not created_raw:
                    continue
                try:
                    created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if created < cutoff:
                    continue

                body = msg.get("body") or ""
                if not body.strip():
                    continue

                # entities.sentiment is null on untagged posts, which is most of them.
                author_tag: Sentiment | None = None
                entities = msg.get("entities") or {}
                sentiment_obj = entities.get("sentiment") or {}
                basic = (sentiment_obj.get("basic") or "").lower()
                if basic == "bullish":
                    author_tag = Sentiment.BULLISH
                elif basic == "bearish":
                    author_tag = Sentiment.BEARISH

                user = msg.get("user") or {}
                msg_id = msg.get("id")
                if not msg_id:
                    continue

                results.append(
                    (
                        RawPost(
                            source=Source.STOCKTWITS,
                            source_id=str(msg_id),
                            text=body,
                            url=f"https://stocktwits.com/message/{msg_id}",
                            created_utc=created,
                            author=user.get("username"),
                            subsource=symbol,
                            engagement=int((msg.get("likes") or {}).get("total", 0) or 0),
                        ),
                        author_tag,
                    )
                )

        tagged = sum(1 for _, tag in results if tag is not None)
        logger.info(
            "StockTwits: fetched %d posts (%d carry an author sentiment tag)", len(results), tagged
        )
        return results

    def status(self) -> str:
        if not self.enabled:
            return "disabled (STOCKTWITS_ENABLED=false)"
        return f"enabled ({len(self.tickers)} symbols/run, no auth required)"
