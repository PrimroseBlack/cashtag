"""Tests for sentiment aggregation.

The dual-denominator design is the thing most likely to be "fixed" by a future
maintainer who assumes the three percentages should sum to 100. These tests pin
the intended behaviour so that change fails loudly instead of silently altering
what every number in the product means.
"""

from __future__ import annotations

import pytest

from cashtag.buzz import aggregate_sentiment


class TestDirectionalSplit:
    def test_bullish_pct_excludes_neutral_from_denominator(self):
        # 10 bullish, 10 bearish, 80 neutral. The people with an opinion are
        # evenly split, so bullish_pct is 50 — NOT 10 (which folding neutral into
        # the denominator would give, and which reads as overwhelmingly bearish).
        split = aggregate_sentiment(bullish=10, bearish=10, neutral=80)
        assert split.bullish_pct == 50.0
        assert split.bearish_pct == 50.0

    def test_unanimous_bullish_is_100_even_when_mostly_neutral(self):
        # The case that makes the design worth defending: opinion is unanimously
        # bullish, and a naive three-way share would report 10% and imply bearish.
        split = aggregate_sentiment(bullish=10, bearish=0, neutral=90)
        assert split.bullish_pct == 100.0
        assert split.bearish_pct == 0.0

    def test_unanimous_bearish_is_zero(self):
        split = aggregate_sentiment(bullish=0, bearish=25, neutral=5)
        assert split.bullish_pct == 0.0
        assert split.bearish_pct == 100.0

    def test_bullish_and_bearish_always_sum_to_100(self):
        for bull, bear in [(1, 99), (50, 50), (3, 1), (7, 13)]:
            split = aggregate_sentiment(bullish=bull, bearish=bear, neutral=5)
            assert split.bullish_pct + split.bearish_pct == pytest.approx(100.0)

    def test_two_thirds_bullish(self):
        split = aggregate_sentiment(bullish=20, bearish=10, neutral=0)
        assert split.bullish_pct == pytest.approx(66.7)
        assert split.bearish_pct == pytest.approx(33.3)


class TestNeutralIsSeparate:
    def test_neutral_pct_uses_all_classified_as_denominator(self):
        # 20 of 100 classified posts are neutral.
        split = aggregate_sentiment(bullish=40, bearish=40, neutral=20)
        assert split.neutral_pct == 20.0

    def test_three_percentages_deliberately_do_not_sum_to_100(self):
        # Pinning the surprising-but-intended behaviour. If someone "fixes" this,
        # this test tells them it was a decision, not an oversight.
        split = aggregate_sentiment(bullish=40, bearish=40, neutral=20)
        total = split.bullish_pct + split.bearish_pct + split.neutral_pct
        assert total == pytest.approx(120.0)
        assert total != pytest.approx(100.0)

    def test_all_neutral_gives_100_neutral_and_null_direction(self):
        split = aggregate_sentiment(bullish=0, bearish=0, neutral=50)
        assert split.neutral_pct == 100.0
        assert split.bullish_pct is None
        assert split.bearish_pct is None


class TestUndefinedDenominators:
    def test_no_opinionated_posts_returns_null_not_fifty(self):
        # Null, not 50. A 50 here would be indistinguishable from a genuinely
        # evenly-split market — the caller could not tell "no data" from "tied".
        split = aggregate_sentiment(bullish=0, bearish=0, neutral=10)
        assert split.bullish_pct is None
        assert split.bearish_pct is None

    def test_nothing_classified_at_all_returns_all_nulls(self):
        split = aggregate_sentiment(bullish=0, bearish=0, neutral=0)
        assert split.bullish_pct is None
        assert split.bearish_pct is None
        assert split.neutral_pct is None

    def test_no_division_by_zero_on_empty_input(self):
        aggregate_sentiment(0, 0, 0, unclassified=0)  # must not raise


class TestRawCountsArePreserved:
    def test_counts_pass_through_untouched(self):
        # The percentages are lossy by construction; the raw counts are the escape
        # hatch for any caller that wants a different denominator.
        split = aggregate_sentiment(bullish=7, bearish=3, neutral=5, unclassified=2)
        assert (split.bullish_count, split.bearish_count, split.neutral_count) == (7, 3, 5)
        assert split.unclassified_count == 2

    def test_unclassified_does_not_affect_any_percentage(self):
        # Unclassified is a data-quality signal, not a label. A backlog must not
        # move the reported sentiment.
        a = aggregate_sentiment(bullish=10, bearish=5, neutral=5, unclassified=0)
        b = aggregate_sentiment(bullish=10, bearish=5, neutral=5, unclassified=500)
        assert a.bullish_pct == b.bullish_pct
        assert a.neutral_pct == b.neutral_pct
        assert b.unclassified_count == 500
