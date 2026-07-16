"""Tests for buzz detection.

These target the pure functions, which is where the methodology actually lives.
The emphasis is on the boundaries and the degenerate inputs — the threshold edges,
the zero-baseline divide, and the two cases (high-volume-no-anomaly,
great-ratio-no-volume) that a single-threshold system gets wrong.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from cashtag.buzz import (
    compute_baseline_avg,
    compute_buzz_score,
    is_buzzing,
    window_bounds,
)
from cashtag.config import (
    BASELINE_DAYS,
    BASELINE_FLOOR,
    BUZZ_MIN_MENTIONS_24H,
    BUZZ_SCORE_THRESHOLD,
    TRAILING_HOURS,
)


class TestComputeBuzzScore:
    def test_score_is_ratio_of_current_to_baseline(self):
        assert compute_buzz_score(mentions_24h=40, baseline_daily_avg=10.0) == 4.0

    def test_score_of_exactly_one_means_perfectly_normal(self):
        assert compute_buzz_score(mentions_24h=10, baseline_daily_avg=10.0) == 1.0

    def test_zero_mentions_scores_zero_not_nan(self):
        assert compute_buzz_score(mentions_24h=0, baseline_daily_avg=10.0) == 0.0

    def test_zero_mentions_and_zero_baseline_is_zero_not_nan(self):
        # 0/0 must not produce nan — nan poisons the sort in list_trending.
        assert compute_buzz_score(mentions_24h=0, baseline_daily_avg=0.0) == 0.0

    def test_zero_baseline_does_not_divide_by_zero(self):
        # The whole point of BASELINE_FLOOR: a brand-new ticker must not score inf
        # and pin itself to the top of the list forever.
        score = compute_buzz_score(mentions_24h=20, baseline_daily_avg=0.0)
        assert score == pytest.approx(20 / BASELINE_FLOOR)
        assert score != float("inf")

    def test_baseline_below_floor_is_clamped_to_floor(self):
        # A baseline of 0.01/day would otherwise yield a 2000x score.
        assert compute_buzz_score(10, 0.01) == compute_buzz_score(10, 0.0)

    def test_baseline_above_floor_is_used_directly(self):
        assert compute_buzz_score(30, 3.0) == 10.0

    def test_score_is_rounded_to_two_places(self):
        assert compute_buzz_score(10, 3.0) == 3.33


class TestComputeBaselineAvg:
    def test_divides_total_by_baseline_days(self):
        assert compute_baseline_avg(BASELINE_DAYS * 7) == 7.0

    def test_zero_history_is_zero_average(self):
        assert compute_baseline_avg(0) == 0.0


class TestIsBuzzing:
    def test_requires_both_thresholds(self):
        assert is_buzzing(mentions_24h=50, buzz_score=5.0) is True

    def test_high_ratio_but_too_few_mentions_is_not_buzzing(self):
        # The RIVN case. 1 -> 12 mentions/day is a 12x ratio and still noise.
        # A ratio-only system flags this and is wrong.
        assert is_buzzing(mentions_24h=12, buzz_score=12.0) is False

    def test_high_volume_but_normal_ratio_is_not_buzzing(self):
        # The TSLA case. 45 mentions is a lot, but it is a Tuesday.
        # An absolute-only system flags this and is wrong.
        assert is_buzzing(mentions_24h=45, buzz_score=1.13) is False

    def test_both_thresholds_failing_is_not_buzzing(self):
        assert is_buzzing(mentions_24h=3, buzz_score=1.0) is False

    def test_exactly_at_both_thresholds_is_buzzing(self):
        # Thresholds are inclusive (>=). Pinning this so a future refactor to
        # strict > is caught rather than silently narrowing the filter.
        assert is_buzzing(BUZZ_MIN_MENTIONS_24H, BUZZ_SCORE_THRESHOLD) is True

    def test_one_below_mention_threshold_is_not_buzzing(self):
        assert is_buzzing(BUZZ_MIN_MENTIONS_24H - 1, BUZZ_SCORE_THRESHOLD) is False

    def test_just_below_score_threshold_is_not_buzzing(self):
        assert is_buzzing(BUZZ_MIN_MENTIONS_24H, BUZZ_SCORE_THRESHOLD - 0.01) is False


class TestWindowBounds:
    def test_baseline_and_trailing_windows_do_not_overlap(self):
        # The most consequential invariant in the methodology. If the trailing 24h
        # were inside its own baseline, a spike would inflate its own denominator
        # and suppress its own score — the detector would be least sensitive
        # exactly when something is happening.
        baseline_start, window_start, now = window_bounds()
        assert baseline_start < window_start < now
        assert now - window_start == timedelta(hours=TRAILING_HOURS)
        assert window_start - baseline_start == timedelta(days=BASELINE_DAYS)

    def test_windows_abut_exactly_with_no_gap(self):
        # A gap would silently discard mentions from the baseline.
        baseline_start, window_start, now = window_bounds()
        assert window_start - baseline_start == timedelta(days=BASELINE_DAYS)
        assert (now - baseline_start) == timedelta(days=BASELINE_DAYS, hours=TRAILING_HOURS)


class TestRealisticScenarios:
    """The scenarios the demo dataset is built to show."""

    def test_meme_stock_squeeze_is_flagged(self):
        # GME: quiet baseline, sudden flood.
        score = compute_buzz_score(95, compute_baseline_avg(8 * BASELINE_DAYS))
        assert score == pytest.approx(11.88, abs=0.01)
        assert is_buzzing(95, score) is True

    def test_megacap_on_a_normal_day_is_not_flagged(self):
        # TSLA: always loud, nothing happening.
        score = compute_buzz_score(45, compute_baseline_avg(40 * BASELINE_DAYS))
        assert is_buzzing(45, score) is False

    def test_tiny_ticker_blip_is_not_flagged(self):
        # RIVN: 1 -> 12/day. Great ratio, still noise.
        score = compute_buzz_score(12, compute_baseline_avg(1 * BASELINE_DAYS))
        assert score >= BUZZ_SCORE_THRESHOLD  # ratio alone would pass
        assert is_buzzing(12, score) is False  # absolute floor rejects it

    def test_high_volume_ticker_with_real_news_is_flagged(self):
        # NVDA: already loud, and now materially louder.
        score = compute_buzz_score(68, compute_baseline_avg(30 * BASELINE_DAYS))
        assert score == pytest.approx(2.27, abs=0.01)
        assert is_buzzing(68, score) is True
