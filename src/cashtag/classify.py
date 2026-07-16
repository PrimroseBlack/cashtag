"""Sentiment classification via the Claude API.

Separate from ingestion by design. Ingestion is I/O-bound and rate-limited by
five different upstreams; classification is a metered spend against one API. If
they shared a step, a Reddit outage would stop labelling and a classifier hiccup
would drop posts on the floor. Split, each retries on its own terms and the
mentions table is the queue between them (`sentiment IS NULL` is the backlog).

Model choice: Haiku 4.5. This is high-volume, low-difficulty 3-way
classification — exactly the shape Haiku is for. Two API details that are easy
to get wrong here:
  - Haiku 4.5 does NOT support the `effort` parameter; passing it errors.
  - Haiku 4.5 DOES support structured outputs, which is what makes the batched
    array response reliable enough to skip hand-parsing.

Batching: posts are packed CLASSIFY_BATCH_SIZE at a time into one request rather
than sent through the Batches API. The Batches API would halve the cost but can
take up to an hour to return, which does not fit a 15-minute ingest loop — labels
would always be a cycle behind. Packing amortizes the instruction prompt across
20 posts and returns inside the same tick.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import (
    CLASSIFY_BATCH_SIZE,
    CLASSIFY_MAX_PER_RUN,
    CLASSIFY_TEXT_CHAR_LIMIT,
    CLASSIFIER_MODEL,
    settings,
)
from .db import Mention, utcnow
from .models import Sentiment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You classify the stance of retail investor social media posts toward the stock(s) they discuss.

Return exactly one label per post:
- "bullish": the author expects the price to RISE, or holds//wants a long position \
(buying, calls, "to the moon", holding through a dip, bought the dip).
- "bearish": the author expects the price to FALL, or holds/wants a short position \
(selling, puts, "it's overvalued", "this is a bubble", calling a top).
- "neutral": no directional view. Questions, news links without commentary, \
educational content, portfolio screenshots without opinion, requests for advice, \
or posts too ambiguous to call.

Rules:
- Judge the AUTHOR's stance, not whether the news is objectively good or bad. \
"Earnings beat but I sold" is bearish.
- Sarcasm and self-deprecation are common on these forums. "Guess I'll be eating \
crayons again" after a loss is still bearish about the outcome, not neutral.
- Loss porn / gain porn without a forward-looking view is neutral.
- If a post takes different sides on different tickers, give the post the label \
matching its DOMINANT stance.
- When genuinely torn, choose "neutral". A wrong directional label is more \
damaging than an abstention.

Return one entry per input post, using the exact index given. Do not skip or \
reorder posts."""


class PostLabel(BaseModel):
    """One classified post."""

    index: int = Field(..., description="The index of the post, exactly as given in the input.")
    sentiment: Sentiment = Field(..., description="bullish, bearish, or neutral.")


class BatchLabels(BaseModel):
    """Structured response for a batch of posts."""

    labels: list[PostLabel] = Field(..., description="One entry per input post.")


def _truncate(text: str, limit: int = CLASSIFY_TEXT_CHAR_LIMIT) -> str:
    """Trim post text before it costs us tokens.

    Directional stance on these forums is essentially always in the opening —
    the title and first paragraph. DD posts run to thousands of words of
    supporting argument that rarely changes the label, and we pay for every one.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _build_batch_prompt(posts: list[tuple[int, str]]) -> str:
    """Render numbered posts into a single user message."""
    lines = []
    for idx, text in posts:
        lines.append(f"--- POST {idx} ---\n{_truncate(text)}")
    return (
        f"Classify each of the following {len(posts)} posts.\n\n"
        + "\n\n".join(lines)
        + "\n\nReturn one label per post, using the exact POST index shown."
    )


def classify_batch(client, posts: list[tuple[int, str]]) -> dict[int, Sentiment]:
    """Classify one batch of (index, text) pairs.

    Args:
        client: An `anthropic.Anthropic` instance.
        posts: (index, text) pairs. Indices need only be unique within the batch.

    Returns:
        Mapping of index -> Sentiment. Indices the model omitted are absent from
        the result rather than defaulted — an unlabelled post stays in the queue
        and gets retried, which is strictly better than silently recording a
        guess as if it were a classification.
    """
    if not posts:
        return {}

    response = client.messages.parse(
        model=CLASSIFIER_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_batch_prompt(posts)}],
        output_format=BatchLabels,
    )

    parsed = response.parsed_output
    if parsed is None:
        logger.warning("Classifier returned no parseable output for batch of %d", len(posts))
        return {}

    valid_indices = {idx for idx, _ in posts}
    labels: dict[int, Sentiment] = {}
    for label in parsed.labels:
        # Guard against hallucinated indices. Structured outputs guarantee the
        # SHAPE of the response, not that the integers correspond to real posts.
        if label.index in valid_indices:
            labels[label.index] = label.sentiment
        else:
            logger.warning("Classifier returned unknown post index %s; dropping", label.index)

    missing = valid_indices - labels.keys()
    if missing:
        logger.warning(
            "Classifier omitted %d/%d posts; will retry next run", len(missing), len(posts)
        )

    return labels


def classify_pending(session: Session, client=None, max_posts: int = CLASSIFY_MAX_PER_RUN) -> int:
    """Label unclassified mentions. Returns the number of mention ROWS updated.

    Grouping: the unit of classification is the POST, not the (post, ticker) row.
    A post mentioning three tickers is classified once and the label is written to
    all three rows. This matches the spec and keeps cost proportional to posts
    rather than to mentions.

    The tradeoff, stated so it is not discovered later: a post like
    "long $GME, short $AMC" gets ONE label applied to both tickers, so one of them
    is wrong. Measured against the corpus this is rare — the large majority of
    posts are single-ticker — but it is a real ceiling on per-ticker accuracy for
    comparison/pairs-trade posts. Lifting it means classifying per (post, ticker),
    which costs roughly 1.2x. See README "Known limitations".
    """
    if client is None:
        if not settings.classifier_configured:
            logger.info("ANTHROPIC_API_KEY not set; skipping classification")
            return 0
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    pending = (
        session.execute(
            select(Mention)
            .where(Mention.sentiment.is_(None))
            .order_by(Mention.created_utc.desc())
            .limit(max_posts)
        )
        .scalars()
        .all()
    )

    if not pending:
        return 0

    # Collapse (post, ticker) rows back to unique posts.
    by_post: dict[tuple[str, str], list[Mention]] = defaultdict(list)
    for mention in pending:
        by_post[(mention.source, mention.source_id)].append(mention)

    post_keys = list(by_post.keys())
    indexed_posts = [(i, by_post[key][0].text) for i, key in enumerate(post_keys)]

    updated_rows = 0
    for start in range(0, len(indexed_posts), CLASSIFY_BATCH_SIZE):
        chunk = indexed_posts[start : start + CLASSIFY_BATCH_SIZE]
        try:
            labels = classify_batch(client, chunk)
        except Exception as exc:
            # One bad batch must not abandon the rest. Unlabelled rows remain
            # NULL and are picked up on the next tick.
            logger.exception("Classification batch failed, continuing: %s", exc)
            continue

        now = utcnow()
        for idx, sentiment in labels.items():
            for mention in by_post[post_keys[idx]]:
                mention.sentiment = sentiment.value
                mention.sentiment_model = CLASSIFIER_MODEL
                mention.classified_at = now
                updated_rows += 1

    session.flush()
    logger.info("Classified %d mention rows across %d posts", updated_rows, len(post_keys))
    return updated_rows
