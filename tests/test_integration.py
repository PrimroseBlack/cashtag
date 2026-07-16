"""End-to-end tests over a real (temporary) database.

The unit tests prove the methodology is right. These prove the wiring is: that
posts survive extraction into storage, that re-polling does not double-count, and
that the two MCP tools return correct, well-formed payloads.
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta

import pytest


@pytest.fixture(scope="module", autouse=True)
def temp_database():
    """Point every module at a scratch SQLite file for the duration.

    Set before importing cashtag modules so the module-level `settings` snapshot
    picks it up. `cashtag.db` caches its engine, so it is reset explicitly too.
    """
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_integration.db")
    os.environ["CASHTAG_DATABASE_URL"] = f"sqlite:///{db_path}"

    import cashtag.config as config_mod
    import cashtag.db as db_mod

    config_mod.settings = config_mod.Settings()
    db_mod.settings = config_mod.settings
    db_mod._engine = None
    db_mod._SessionLocal = None

    db_mod.init_db()
    yield
    db_mod.reset_db()


@pytest.fixture
def clean_db():
    from cashtag.db import reset_db

    reset_db()
    yield


def _make_post(source_id: str, text: str, hours_ago: float, engagement: int = 0):
    from cashtag.db import utcnow
    from cashtag.models import Source
    from cashtag.sources.base import RawPost

    return RawPost(
        source=Source.REDDIT,
        source_id=source_id,
        text=text,
        url=f"https://reddit.com/r/wallstreetbets/comments/{source_id}",
        created_utc=utcnow() - timedelta(hours=hours_ago),
        author="tester",
        subsource="wallstreetbets",
        engagement=engagement,
    )


class TestStorePosts:
    def test_writes_one_row_per_ticker_in_a_post(self, clean_db):
        # A post naming two tickers is two mentions, not one.
        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        with session_scope() as session:
            written = store_posts(session, [_make_post("p1", "$GME and $AMC squeeze", 1)])
        assert written == 2

    def test_skips_posts_with_no_tickers(self, clean_db):
        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        with session_scope() as session:
            written = store_posts(session, [_make_post("p1", "I love the stock market", 1)])
        assert written == 0

    def test_repolling_the_same_post_does_not_double_count(self, clean_db):
        # Every poll re-reads an overlapping window on purpose. Duplicate
        # suppression is the hot path, not an error case — if this regresses,
        # every buzz score inflates with each tick and the whole product lies.
        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        post = _make_post("p1", "$GME to the moon", 1)
        with session_scope() as session:
            first = store_posts(session, [post])
        with session_scope() as session:
            second = store_posts(session, [post])

        assert first == 1
        assert second == 0

    def test_one_duplicate_does_not_discard_the_rest_of_the_batch(self, clean_db):
        # The savepoint-per-row behaviour. Without it, a single dupe rolls back
        # every post in the tick.
        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        post_a = _make_post("dup", "$GME squeeze", 1)
        with session_scope() as session:
            store_posts(session, [post_a])

        with session_scope() as session:
            written = store_posts(
                session,
                [post_a, _make_post("new1", "$TSLA calls", 1), _make_post("new2", "$NVDA up", 1)],
            )
        assert written == 2  # the two new ones survived the duplicate

    def test_marks_synthetic_rows(self, clean_db):
        from sqlalchemy import select

        from cashtag.db import Mention, session_scope
        from cashtag.worker import store_posts

        with session_scope() as session:
            store_posts(session, [_make_post("p1", "$GME up", 1)], is_synthetic=True)
        with session_scope() as session:
            row = session.execute(select(Mention)).scalars().first()
        assert row.is_synthetic is True

    def test_persists_author_sentiment_tags(self, clean_db):
        from sqlalchemy import select

        from cashtag.db import Mention, session_scope
        from cashtag.models import Sentiment
        from cashtag.worker import store_posts

        post = _make_post("p1", "$GME up", 1)
        with session_scope() as session:
            store_posts(
                session,
                [post],
                author_tags={("reddit", "p1"): Sentiment.BULLISH},
            )
        with session_scope() as session:
            row = session.execute(select(Mention)).scalars().first()

        assert row.author_sentiment == "bullish"
        # The classifier must still label it blind — that's what makes the
        # author tag usable as ground truth.
        assert row.sentiment is None


class TestBuzzTools:
    def _seed_spike(self):
        """15 baseline days at 1/day for GME, then 30 mentions in the last 24h."""
        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        posts = []
        for day in range(1, 15):
            posts.append(_make_post(f"base-{day}", "$GME discussion", hours_ago=day * 24 + 2))
        for i in range(30):
            posts.append(
                _make_post(f"spike-{i}", "$GME to the moon", hours_ago=i * 0.5, engagement=i * 10)
            )
        with session_scope() as session:
            store_posts(session, posts)

    def test_list_trending_surfaces_a_spike(self, clean_db):
        from cashtag.models import ListTrendingInput
        from cashtag.server import buzz_list_trending

        self._seed_spike()
        result = buzz_list_trending(ListTrendingInput(limit=10))

        assert result.count == 1
        top = result.tickers[0]
        assert top.ticker == "GME"
        assert top.mention_count_24h == 30
        assert top.buzz_score > 2.0

    def test_list_trending_echoes_its_own_criteria(self, clean_db):
        # So Claude can explain WHY something is listed without reading the source.
        from cashtag.models import ListTrendingInput
        from cashtag.server import buzz_list_trending

        self._seed_spike()
        result = buzz_list_trending(ListTrendingInput(limit=10))
        assert "15" in result.criteria and "2.0" in result.criteria

    def test_list_trending_returns_sample_urls(self, clean_db):
        from cashtag.models import ListTrendingInput
        from cashtag.server import buzz_list_trending

        self._seed_spike()
        result = buzz_list_trending(ListTrendingInput(limit=10))
        urls = result.tickers[0].sample_post_urls
        assert 1 <= len(urls) <= 3
        assert all(u.startswith("https://") for u in urls)

    def test_empty_store_returns_a_note_not_an_error(self, clean_db):
        # "Nothing is buzzing" is a real answer on a quiet tape.
        from cashtag.models import ListTrendingInput
        from cashtag.server import buzz_list_trending

        result = buzz_list_trending(ListTrendingInput(limit=10))
        assert result.count == 0
        assert result.notes

    def test_get_ticker_works_for_non_buzzing_tickers(self, clean_db):
        from cashtag.models import GetTickerInput
        from cashtag.server import buzz_get_ticker

        from cashtag.db import session_scope
        from cashtag.worker import store_posts

        with session_scope() as session:
            store_posts(session, [_make_post(f"q-{i}", "$TSLA thoughts?", 1) for i in range(3)])

        result = buzz_get_ticker(GetTickerInput(ticker="TSLA"))
        assert result.ticker == "TSLA"
        assert result.mention_count_24h == 3
        assert result.is_buzzing is False  # only 3 mentions

    def test_get_ticker_for_unknown_ticker_returns_zeros_and_a_note(self, clean_db):
        from cashtag.models import GetTickerInput
        from cashtag.server import buzz_get_ticker

        result = buzz_get_ticker(GetTickerInput(ticker="AAPL"))
        assert result.mention_count_24h == 0
        assert result.is_buzzing is False
        assert any("No mentions" in n for n in result.notes)

    def test_get_ticker_normalizes_dollar_prefix_and_case(self, clean_db):
        from cashtag.models import GetTickerInput
        from cashtag.server import buzz_get_ticker

        assert buzz_get_ticker(GetTickerInput(ticker="$gme")).ticker == "GME"

    def test_get_ticker_trend_is_seven_zero_filled_days(self, clean_db):
        from cashtag.models import GetTickerInput
        from cashtag.server import buzz_get_ticker

        self._seed_spike()
        result = buzz_get_ticker(GetTickerInput(ticker="GME"))
        assert len(result.trend_7d) == 7
        # Quiet days must appear as zeros, not be omitted — a caller plotting
        # this should see the gap, not a compressed line implying continuity.
        assert all(isinstance(p.mention_count, int) for p in result.trend_7d)

    def test_get_ticker_source_breakdown(self, clean_db):
        from cashtag.models import GetTickerInput
        from cashtag.server import buzz_get_ticker

        self._seed_spike()
        result = buzz_get_ticker(GetTickerInput(ticker="GME"))
        assert len(result.source_breakdown) == 1
        assert result.source_breakdown[0].source.value == "reddit"
        assert result.source_breakdown[0].mention_count == 30

    def test_synthetic_data_is_flagged_in_tool_output(self, clean_db):
        # The safety-critical path: fake data must never read as real signal.
        from cashtag.db import session_scope
        from cashtag.models import ListTrendingInput
        from cashtag.server import buzz_list_trending
        from cashtag.worker import store_posts

        posts = [_make_post(f"base-{d}", "$GME chat", d * 24 + 2) for d in range(1, 15)]
        posts += [_make_post(f"s-{i}", "$GME moon", i * 0.5) for i in range(30)]
        with session_scope() as session:
            store_posts(session, posts, is_synthetic=True)

        result = buzz_list_trending(ListTrendingInput(limit=5))
        assert result.tickers[0].is_synthetic is True
        assert any("SYNTHETIC" in n for n in result.notes)


class TestInputValidation:
    def test_rejects_non_alphabetic_ticker(self):
        from pydantic import ValidationError

        from cashtag.models import GetTickerInput

        with pytest.raises(ValidationError):
            GetTickerInput(ticker="TSLA240C")

    def test_rejects_empty_ticker(self):
        from pydantic import ValidationError

        from cashtag.models import GetTickerInput

        with pytest.raises(ValidationError):
            GetTickerInput(ticker="$")

    def test_rejects_out_of_range_limit(self):
        from pydantic import ValidationError

        from cashtag.models import ListTrendingInput

        with pytest.raises(ValidationError):
            ListTrendingInput(limit=0)
        with pytest.raises(ValidationError):
            ListTrendingInput(limit=101)

    def test_rejects_unknown_fields(self):
        from pydantic import ValidationError

        from cashtag.models import ListTrendingInput

        with pytest.raises(ValidationError):
            ListTrendingInput(limit=10, sneaky_field="x")
