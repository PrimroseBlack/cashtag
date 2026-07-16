"""Reddit ingestion via the official API (OAuth, PRAW).

Access model: official API, OAuth2 client-credentials, free non-commercial tier
at 100 requests/minute. This is the only one of the four sources with a real,
free, sanctioned read API — which is exactly why the build order validates the
whole pipeline here before adding anything else.

Two operational notes that bite people:

1. The User-Agent is not cosmetic. Reddit enforces a descriptive UA with contact
   info and will 429/403 a generic or absent one. Format:
   `python:cashtag:0.1.0 (by /u/username)`.

2. The free non-commercial tier requires a separate approval form beyond
   registering the app, and review can take 2-4 weeks. Registering the app at
   reddit.com/prefs/apps gets you credentials that work at a lower limit; the
   free-tier form is what sanctions ongoing non-commercial use. Start it early —
   it is the long pole in this build.

We read `.new()` rather than `.hot()`: buzz detection needs a complete census of
what was posted in a window, and `.hot()` is a rank, not a window — it would
oversample popular posts and undersample exactly the early, low-engagement posts
that make a spike detectable before it is obvious.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..config import REDDIT_POSTS_PER_SUBREDDIT, settings
from ..models import Source
from .base import IngestionSource, RawPost

logger = logging.getLogger(__name__)

#: Only ingest posts newer than this. The baseline needs 15 days of history, but
#: each poll only needs to cover the gap since the last poll plus slack.
LOOKBACK_HOURS = 26


class RedditSource(IngestionSource):
    """Polls a configurable list of subreddits for new submissions."""

    def __init__(self, subreddits: list[str] | None = None, client=None):
        self.subreddits = subreddits or settings.subreddits
        self._client = client
        self.enabled = settings.reddit_configured or client is not None

        if not self.enabled:
            logger.info(
                "Reddit disabled: REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set. "
                "Register a script app at https://www.reddit.com/prefs/apps"
            )

    @property
    def name(self) -> Source:
        return Source.REDDIT

    def _get_client(self):
        if self._client is None:
            import praw

            self._client = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
                check_for_async=False,
            )
            # Read-only: we never post, and this avoids PRAW attempting a
            # password grant it has no credentials for.
            self._client.read_only = True
        return self._client

    def fetch(self) -> list[RawPost]:
        """Fetch recent submissions across the configured subreddits.

        A failure in one subreddit (private, banned, renamed, rate-limited) is
        logged and skipped — the other subreddits still produce data for this tick.
        """
        if not self.enabled:
            return []

        try:
            reddit = self._get_client()
        except Exception as exc:
            logger.exception("Could not initialize Reddit client: %s", exc)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
        posts: list[RawPost] = []

        for subreddit_name in self.subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for submission in subreddit.new(limit=REDDIT_POSTS_PER_SUBREDDIT):
                    created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                    if created < cutoff:
                        # .new() is reverse-chronological, so the first old post
                        # means every remaining post is older. Stop, don't continue —
                        # saves the rest of the pagination.
                        break

                    body = getattr(submission, "selftext", "") or ""
                    text = f"{submission.title}\n\n{body}".strip()

                    posts.append(
                        RawPost(
                            source=Source.REDDIT,
                            source_id=submission.id,
                            text=text,
                            url=f"https://reddit.com{submission.permalink}",
                            created_utc=created,
                            author=str(submission.author) if submission.author else None,
                            subsource=subreddit_name,
                            engagement=int(getattr(submission, "score", 0) or 0),
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to fetch r/%s: %s", subreddit_name, exc)
                continue

        logger.info(
            "Reddit: fetched %d posts across %d subreddits", len(posts), len(self.subreddits)
        )
        return posts

    def status(self) -> str:
        if not self.enabled:
            return "disabled (REDDIT_CLIENT_ID/SECRET not set)"
        return f"enabled ({len(self.subreddits)} subreddits: {', '.join(self.subreddits)})"
