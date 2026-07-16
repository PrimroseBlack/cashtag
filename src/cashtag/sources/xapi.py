"""X/Twitter ingestion via the official X API — metered, pay-per-use.

Access model: official X API v2 recent search. As of Feb 2026 there is no free
tier; reads bill at roughly $0.005 each. That single fact drives every design
decision in this file, and none of them apply to the other four sources:

1. HARD BUDGET CAP, CHECKED BEFORE EVERY CALL.
   `X_MONTHLY_BUDGET_USD` is enforced against the `api_spend` ledger in the
   database, not an in-process counter. This matters: an in-memory counter resets
   on every deploy, restart, and crash-loop — on a platform that restarts
   containers, that is an unbounded bill wearing a budget's clothing. The check
   is also pre-flight, not post-hoc: we refuse to make the call that would
   exceed, rather than notice afterwards.

2. CURSORED READS (`since_id`), PERSISTED.
   Every re-fetch of a tweet we already have is money spent to learn nothing.
   The cursor per query lives in `source_cursors` so it survives restarts too.

3. CASHTAG SEARCH, NOT KEYWORD SEARCH.
   We query `$TSLA OR $GME ...` rather than free text. Cashtag search is both far
   more precise (X resolves the entity) and far cheaper, because we are not
   paying to read and then discard thousands of irrelevant results.

The cap fails CLOSED: when the budget is exhausted, this source returns nothing
and logs a warning. The connector keeps working on its other sources. An
over-budget bill is a worse outcome than a missing source.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy.orm import Session

from ..config import X_COST_PER_READ_USD, X_MAX_RESULTS_PER_QUERY, settings
from ..db import month_to_date_spend, record_spend
from ..models import Source
from .base import IngestionSource, RawPost

logger = logging.getLogger(__name__)

X_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

#: Cashtags per query. X caps query length, and batching symbols into one OR
#: query is what keeps the read count proportional to *tickers* rather than
#: to tickers x polls.
CASHTAGS_PER_QUERY = 20


class XSource(IngestionSource):
    """Polls X recent-search for cashtag mentions, under a hard monthly budget."""

    def __init__(self, tickers: list[str], session_factory=None, client=None):
        self.tickers = tickers
        self._client = client
        self._session_factory = session_factory
        self.enabled = settings.x_enabled and bool(settings.x_bearer_token)

        if settings.x_enabled and not settings.x_bearer_token:
            logger.warning("X_ENABLED=true but X_BEARER_TOKEN is not set; X source disabled")

    @property
    def name(self) -> Source:
        return Source.X

    def _budget_remaining(self, session: Session) -> float:
        """USD left in this UTC month's budget."""
        spent = month_to_date_spend(session, Source.X.value)
        return settings.x_monthly_budget_usd - spent

    def _affordable_reads(self, session: Session) -> int:
        """How many reads the remaining budget permits. Zero means stop."""
        remaining = self._budget_remaining(session)
        if remaining <= 0:
            return 0
        return int(remaining / X_COST_PER_READ_USD)

    def _build_queries(self) -> list[str]:
        """Batch tickers into cashtag OR-queries.

        `-is:retweet` is not politeness — retweets are duplicate text we would pay
        full price to ingest and then dedupe away.
        """
        queries = []
        for start in range(0, len(self.tickers), CASHTAGS_PER_QUERY):
            chunk = self.tickers[start : start + CASHTAGS_PER_QUERY]
            cashtags = " OR ".join(f"${t}" for t in chunk)
            queries.append(f"({cashtags}) -is:retweet lang:en")
        return queries

    def _get_cursor(self, session: Session, query_key: str) -> str | None:
        from ..db import SourceCursor

        row = session.get(SourceCursor, (Source.X.value, query_key))
        return row.cursor if row else None

    def _set_cursor(self, session: Session, query_key: str, cursor: str) -> None:
        from ..db import SourceCursor

        row = session.get(SourceCursor, (Source.X.value, query_key))
        if row:
            row.cursor = cursor
            row.updated_at = datetime.now(timezone.utc)
        else:
            session.add(
                SourceCursor(
                    source=Source.X.value,
                    query_key=query_key,
                    cursor=cursor,
                    updated_at=datetime.now(timezone.utc),
                )
            )

    def fetch(self) -> list[RawPost]:
        """Fetch new cashtag mentions, stopping the moment the budget is exhausted."""
        if not self.enabled:
            return []
        if self._session_factory is None:
            logger.error("XSource requires a session_factory to enforce its budget; skipping")
            return []

        posts: list[RawPost] = []

        with self._session_factory() as session:
            affordable = self._affordable_reads(session)
            if affordable <= 0:
                logger.warning(
                    "X monthly budget of $%.2f is exhausted; skipping X ingestion until next month. "
                    "Other sources are unaffected.",
                    settings.x_monthly_budget_usd,
                )
                return []

            queries = self._build_queries()
            headers = {"Authorization": f"Bearer {settings.x_bearer_token}"}
            client = self._client or httpx.Client(timeout=30.0)

            for query in queries:
                # Re-check before EVERY call: earlier queries in this same loop
                # have already spent, and the cap is a cap, not a starting budget.
                affordable = self._affordable_reads(session)
                if affordable <= 0:
                    logger.warning(
                        "X budget exhausted mid-run; stopping after %d posts", len(posts)
                    )
                    break

                page_size = min(X_MAX_RESULTS_PER_QUERY, affordable)
                if page_size < 10:
                    # X requires max_results >= 10. Less budget than that means done.
                    logger.warning("X budget below one minimum-size page; stopping")
                    break

                params = {
                    "query": query,
                    "max_results": page_size,
                    "tweet.fields": "created_at,public_metrics,author_id",
                }
                since_id = self._get_cursor(session, query)
                if since_id:
                    params["since_id"] = since_id

                try:
                    response = client.get(X_SEARCH_URL, headers=headers, params=params)
                    response.raise_for_status()
                    payload = response.json()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        logger.warning("X rate limit hit; stopping this run")
                        break
                    logger.warning("X search failed (%s); skipping query", exc.response.status_code)
                    continue
                except Exception as exc:
                    logger.warning("X search error: %s", exc)
                    continue

                tweets = payload.get("data", []) or []

                # Bill what we actually received. Recording before parsing would
                # be safer against crashes, but X bills per returned tweet — so
                # this matches the invoice.
                if tweets:
                    record_spend(session, Source.X.value, len(tweets), X_COST_PER_READ_USD)

                newest_id = payload.get("meta", {}).get("newest_id")
                if newest_id:
                    self._set_cursor(session, query, newest_id)

                for tweet in tweets:
                    created_raw = tweet.get("created_at")
                    if not created_raw:
                        continue
                    created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    metrics = tweet.get("public_metrics", {}) or {}
                    engagement = int(metrics.get("like_count", 0)) + int(
                        metrics.get("retweet_count", 0)
                    )
                    posts.append(
                        RawPost(
                            source=Source.X,
                            source_id=str(tweet["id"]),
                            text=tweet.get("text", ""),
                            url=f"https://x.com/i/web/status/{tweet['id']}",
                            created_utc=created,
                            author=tweet.get("author_id"),
                            engagement=engagement,
                        )
                    )

            session.commit()

            spent = month_to_date_spend(session, Source.X.value)
            logger.info(
                "X: fetched %d posts. Month-to-date spend $%.2f of $%.2f budget.",
                len(posts),
                spent,
                settings.x_monthly_budget_usd,
            )

        return posts

    def status(self) -> str:
        if not settings.x_enabled:
            return "disabled (X_ENABLED=false)"
        if not settings.x_bearer_token:
            return "disabled (X_BEARER_TOKEN not set)"
        if self._session_factory is None:
            return "enabled (budget state unavailable)"
        with self._session_factory() as session:
            spent = month_to_date_spend(session, Source.X.value)
        return (
            f"enabled (${spent:.2f} / ${settings.x_monthly_budget_usd:.2f} spent this month, "
            f"~${X_COST_PER_READ_USD}/read)"
        )
