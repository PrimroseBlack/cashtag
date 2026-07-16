"""Instagram and TikTok ingestion via Bright Data structured data feeds.

Why these two share a file and a vendor
---------------------------------------
Neither platform has a usable public search API, for different reasons:

- Instagram: the Graph API only reaches accounts you own or manage. There is no
  sanctioned public hashtag/caption search for third parties. Full stop.
- TikTok: a Research API exists, but access is gated to approved academic and
  nonprofit researchers. A portfolio project does not qualify, and pretending
  otherwise in an interview would be worse than the gap.

So both go through a scraping vendor. That is a real, defensible engineering
decision — and it is also why these are the LOWEST-CONFIDENCE sources here:

1. Async, not real-time. Bright Data's collection model is trigger -> poll ->
   fetch, with minutes of latency. These sources are always a cycle behind.
2. Caption-only. We see the caption, not the video or image. A TikTok whose
   entire thesis is spoken aloud reads as an empty caption with three emojis —
   sentiment classification on that is close to guessing.
3. Lower volume and worse recall. Hashtag coverage is a sample, not a census,
   which breaks the census assumption the buzz baseline depends on.

Consequence, made explicit rather than buried: these sources contribute to
mention counts but should be read as directional colour, not signal. If Instagram
and TikTok are the `top_source` for a ticker, treat that ticker with suspicion —
it is more likely a coverage artifact than a real crowd.

Dataset IDs come from your Bright Data console (Datasets -> the marketplace
dataset you subscribed to). They are per-account, so they are configuration, not
constants.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from ..config import BRIGHTDATA_POST_LIMIT, settings
from ..models import Source
from .base import IngestionSource, RawPost

logger = logging.getLogger(__name__)

TRIGGER_URL = "https://api.brightdata.com/datasets/v3/trigger"
SNAPSHOT_URL = "https://api.brightdata.com/datasets/v3/snapshot"
PROGRESS_URL = "https://api.brightdata.com/datasets/v3/progress"

#: Per-account; read them from your Bright Data console and set in the env.
INSTAGRAM_DATASET_ID = os.getenv("BRIGHTDATA_INSTAGRAM_DATASET_ID", "")
TIKTOK_DATASET_ID = os.getenv("BRIGHTDATA_TIKTOK_DATASET_ID", "")

#: Cashtag hashtags do not exist on these platforms the way they do on X —
#: #TSLA is how the ticker appears, not $TSLA.
DEFAULT_HASHTAGS = ["stocks", "stockmarket", "investing", "trading", "wallstreetbets"]

POLL_INTERVAL_SEC = 10
POLL_TIMEOUT_SEC = 180


class BrightDataSource(IngestionSource):
    """Base for the two Bright Data-backed sources. Not used directly."""

    dataset_id: str = ""
    _source: Source

    def __init__(self, hashtags: list[str] | None = None, client=None):
        self.hashtags = hashtags or DEFAULT_HASHTAGS
        self._client = client
        self.enabled = (
            settings.brightdata_enabled
            and bool(settings.brightdata_api_key)
            and bool(self.dataset_id)
        )
        if settings.brightdata_enabled and not self.dataset_id:
            logger.warning(
                "%s enabled but no dataset ID configured; set the dataset ID env var "
                "from your Bright Data console",
                self._source.value,
            )

    @property
    def name(self) -> Source:
        return self._source

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.brightdata_api_key}",
            "Content-Type": "application/json",
        }

    def _trigger(self, client: httpx.Client) -> str | None:
        """Kick off a collection job. Returns a snapshot_id."""
        payload = [
            {"search_keyword": tag, "num_of_posts": BRIGHTDATA_POST_LIMIT} for tag in self.hashtags
        ]
        try:
            response = client.post(
                TRIGGER_URL,
                headers=self._headers(),
                params={"dataset_id": self.dataset_id, "include_errors": "true"},
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("snapshot_id")
        except Exception as exc:
            logger.warning("%s: Bright Data trigger failed: %s", self._source.value, exc)
            return None

    def _await_snapshot(self, client: httpx.Client, snapshot_id: str) -> list[dict] | None:
        """Poll until the job is ready, then fetch it.

        Bounded by POLL_TIMEOUT_SEC. A collection that has not finished inside
        that window is abandoned for this tick rather than blocking the worker —
        the next tick will trigger a fresh one. These are the lowest-value
        sources; they do not get to hold up the pipeline.
        """
        deadline = time.monotonic() + POLL_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                progress = client.get(f"{PROGRESS_URL}/{snapshot_id}", headers=self._headers())
                progress.raise_for_status()
                status = progress.json().get("status")
                if status == "ready":
                    break
                if status == "failed":
                    logger.warning("%s: Bright Data job failed", self._source.value)
                    return None
            except Exception as exc:
                logger.warning("%s: progress poll error: %s", self._source.value, exc)
                return None
            time.sleep(POLL_INTERVAL_SEC)
        else:
            logger.warning(
                "%s: Bright Data job did not finish in %ss; abandoning this tick",
                self._source.value,
                POLL_TIMEOUT_SEC,
            )
            return None

        try:
            snapshot = client.get(
                f"{SNAPSHOT_URL}/{snapshot_id}",
                headers=self._headers(),
                params={"format": "json"},
            )
            snapshot.raise_for_status()
            data = snapshot.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("%s: snapshot fetch failed: %s", self._source.value, exc)
            return None

    def _parse(self, records: list[dict]) -> list[RawPost]:
        raise NotImplementedError

    def fetch(self) -> list[RawPost]:
        if not self.enabled:
            return []
        client = self._client or httpx.Client(timeout=60.0)

        snapshot_id = self._trigger(client)
        if not snapshot_id:
            return []

        records = self._await_snapshot(client, snapshot_id)
        if not records:
            return []

        posts = self._parse(records)
        logger.info("%s: fetched %d posts", self._source.value, len(posts))
        return posts

    def status(self) -> str:
        if not settings.brightdata_enabled:
            return "disabled (BRIGHTDATA_ENABLED=false)"
        if not settings.brightdata_api_key:
            return "disabled (BRIGHTDATA_API_KEY not set)"
        if not self.dataset_id:
            return "disabled (dataset ID not configured)"
        return f"enabled (hashtags: {', '.join(self.hashtags)}) — low-confidence source"


def _recent_cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=26)


def _parse_ts(raw) -> datetime | None:
    """Best-effort timestamp parse. Scraped feeds are not consistent about format."""
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class InstagramSource(BrightDataSource):
    """Instagram hashtag/caption search via Bright Data."""

    _source = Source.INSTAGRAM
    dataset_id = INSTAGRAM_DATASET_ID

    def _parse(self, records: list[dict]) -> list[RawPost]:
        cutoff = _recent_cutoff()
        posts = []
        for rec in records:
            created = _parse_ts(rec.get("date_posted") or rec.get("timestamp"))
            if not created or created < cutoff:
                continue
            caption = rec.get("caption") or rec.get("description") or ""
            if not caption.strip():
                # Caption-only source: no caption means nothing to extract or classify.
                continue
            post_id = rec.get("post_id") or rec.get("id")
            if not post_id:
                continue
            posts.append(
                RawPost(
                    source=Source.INSTAGRAM,
                    source_id=str(post_id),
                    text=caption,
                    url=rec.get("url") or f"https://instagram.com/p/{post_id}",
                    created_utc=created,
                    author=rec.get("user_username") or rec.get("owner_username"),
                    subsource=rec.get("hashtag"),
                    engagement=int(rec.get("likes") or 0),
                )
            )
        return posts


class TikTokSource(BrightDataSource):
    """TikTok caption/hashtag search via Bright Data."""

    _source = Source.TIKTOK
    dataset_id = TIKTOK_DATASET_ID

    def _parse(self, records: list[dict]) -> list[RawPost]:
        cutoff = _recent_cutoff()
        posts = []
        for rec in records:
            created = _parse_ts(rec.get("create_time") or rec.get("date_posted"))
            if not created or created < cutoff:
                continue
            caption = rec.get("description") or rec.get("caption") or ""
            if not caption.strip():
                continue
            post_id = rec.get("post_id") or rec.get("id")
            if not post_id:
                continue
            posts.append(
                RawPost(
                    source=Source.TIKTOK,
                    source_id=str(post_id),
                    text=caption,
                    url=rec.get("url") or f"https://tiktok.com/@/video/{post_id}",
                    created_utc=created,
                    author=rec.get("profile_username") or rec.get("author"),
                    subsource=rec.get("hashtag"),
                    engagement=int(rec.get("digg_count") or rec.get("likes") or 0),
                )
            )
        return posts
