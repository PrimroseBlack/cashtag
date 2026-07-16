"""Tests for the cold-start / warm-up guard.

Regression tests for a real defect found on the first live ingest. With one tick
of StockTwits data in the store, every ticker had a baseline of 0, which floored
to BASELINE_FLOOR and made buzz_score = 2x the raw count. The tool reported 18
tickers "buzzing" and put AAPL — the most-discussed stock in the world, on an
ordinary Tuesday — at 96x normal.

Nothing threw. Nothing looked broken. The output was just confidently wrong,
which is the worst failure mode a data product has.
"""

from __future__ import annotations

import pytest

from cashtag.buzz import compute_baseline_avg, is_warming_up, warmup_note
from cashtag.config import BASELINE_DAYS, MIN_BASELINE_COVERAGE_DAYS


class TestIsWarmingUp:
    def test_empty_store_is_warming_up(self):
        assert is_warming_up(0.0) is True

    def test_one_day_of_history_is_warming_up(self):
        assert is_warming_up(1.0) is True

    def test_just_below_threshold_is_warming_up(self):
        assert is_warming_up(MIN_BASELINE_COVERAGE_DAYS - 0.1) is True

    def test_exactly_at_threshold_is_not_warming_up(self):
        assert is_warming_up(MIN_BASELINE_COVERAGE_DAYS) is False

    def test_full_baseline_is_not_warming_up(self):
        assert is_warming_up(float(BASELINE_DAYS)) is False


class TestCoverageAwareBaseline:
    def test_partial_coverage_divides_by_observed_days_not_assumed_days(self):
        # 30 mentions over 3 observed days is 10/day, not 30/14 = 2.14/day.
        # Getting this wrong understates the baseline ~4.7x and inflates every
        # score by the same factor.
        assert compute_baseline_avg(30, coverage_days=3.0) == 10.0
        assert compute_baseline_avg(30, coverage_days=BASELINE_DAYS) == pytest.approx(
            2.143, abs=0.01
        )

    def test_defaults_to_full_baseline_window(self):
        # Back-compat: existing callers that don't pass coverage keep old behaviour.
        assert compute_baseline_avg(BASELINE_DAYS * 7) == 7.0

    def test_coverage_below_one_day_is_clamped_to_one(self):
        # Otherwise a 6-hour-old store multiplies the rate 4x.
        assert compute_baseline_avg(10, coverage_days=0.25) == 10.0
        assert compute_baseline_avg(10, coverage_days=0.0) == 10.0

    def test_coverage_above_window_is_clamped_to_window(self):
        assert compute_baseline_avg(140, coverage_days=999.0) == 10.0


class TestIsProvisional:
    def test_near_full_coverage_is_not_flagged_provisional(self):
        # Regression: real ingestion never lands on an exact day boundary, so a
        # strict `coverage < BASELINE_DAYS` fired at 13.96 days and printed
        # "only 14.0 of 14 days observed" — a warning that reads as a bug.
        from cashtag.buzz import is_provisional

        assert is_provisional(13.96) is False

    def test_full_coverage_is_not_provisional(self):
        from cashtag.buzz import is_provisional

        assert is_provisional(float(BASELINE_DAYS)) is False

    def test_genuinely_partial_coverage_is_provisional(self):
        from cashtag.buzz import is_provisional

        assert is_provisional(5.0) is True

    def test_warming_up_is_not_also_reported_as_provisional(self):
        # The two notes are mutually exclusive; emitting both would contradict.
        from cashtag.buzz import is_provisional

        assert is_provisional(1.0) is False
        assert is_warming_up(1.0) is True

    def test_tolerance_boundary(self):
        from cashtag.buzz import is_provisional
        from cashtag.config import PROVISIONAL_TOLERANCE_DAYS

        assert is_provisional(BASELINE_DAYS - PROVISIONAL_TOLERANCE_DAYS) is False
        assert is_provisional(BASELINE_DAYS - PROVISIONAL_TOLERANCE_DAYS - 0.1) is True


class TestWarmupNote:
    def test_note_states_the_actual_coverage(self):
        assert "1.5" in warmup_note(1.5)

    def test_note_says_counts_are_still_usable(self):
        # The distinction that matters: mention counts are correct from tick one.
        # Only the comparison-to-normal needs history.
        note = warmup_note(0.0).lower()
        assert "buzz_get_ticker" in note or "counts" in note


class TestColdStartRegression:
    """The exact scenario from the first live run."""

    def test_the_aapl_96x_bug(self):
        # 48 real AAPL mentions, no history. The old code returned:
        #   baseline 0.0 -> floored to 0.5 -> score 96.0 -> "BUZZING"
        # The score computation itself is still reachable and still does this,
        # which is correct in isolation — the guard is what stops it being
        # reported as a finding.
        from cashtag.buzz import compute_buzz_score, is_buzzing

        naive_score = compute_buzz_score(48, compute_baseline_avg(0, coverage_days=0.0))
        assert naive_score == 96.0
        assert is_buzzing(48, naive_score) is True  # would have been flagged

        # The guard is what prevents it from ever being surfaced.
        assert is_warming_up(0.0) is True

    def test_list_trending_returns_nothing_on_a_cold_store(self, tmp_path, monkeypatch):
        # End-to-end: no history means no trending list, at any volume.
        import os

        monkeypatch.setenv("CASHTAG_DATABASE_URL", f"sqlite:///{tmp_path}/warmup.db")

        import cashtag.config as config_mod
        import cashtag.db as db_mod

        config_mod.settings = config_mod.Settings()
        db_mod.settings = config_mod.settings
        db_mod._engine = None
        db_mod._SessionLocal = None
        db_mod.reset_db()

        from datetime import timedelta

        from cashtag.buzz import baseline_coverage_days, list_trending
        from cashtag.db import Mention, session_scope, utcnow

        now = utcnow()
        with session_scope() as session:
            # 50 mentions, all within the last few hours — exactly what one tick
            # against a live source produces.
            for i in range(50):
                session.add(
                    Mention(
                        ticker="AAPL",
                        source="stocktwits",
                        source_id=f"cold-{i}",
                        text="$AAPL discussion",
                        url=f"https://stocktwits.com/message/{i}",
                        created_utc=now - timedelta(hours=i % 12),
                        ingested_at=now,
                    )
                )

        with session_scope() as session:
            assert baseline_coverage_days(session) == 0.0
            assert list_trending(session, limit=20) == []

        os.environ.pop("CASHTAG_DATABASE_URL", None)


class TestSeededStoreGuard:
    """The worker must not ingest real data into a seeded store."""

    def test_refuses_to_start_when_synthetic_rows_present(self, tmp_path, monkeypatch):
        # Mixing fabricated baselines with real mentions scores a real spike
        # against an invented normal. Nothing errors — the numbers are just wrong.
        import os

        import pytest as _pytest

        monkeypatch.setenv("CASHTAG_DATABASE_URL", f"sqlite:///{tmp_path}/seeded.db")

        import cashtag.config as config_mod
        import cashtag.db as db_mod

        config_mod.settings = config_mod.Settings()
        db_mod.settings = config_mod.settings
        db_mod._engine = None
        db_mod._SessionLocal = None
        db_mod.reset_db()

        from cashtag.db import Mention, session_scope, utcnow
        from cashtag.worker import assert_store_not_seeded

        # Clean store: allowed.
        assert_store_not_seeded()

        with session_scope() as session:
            session.add(
                Mention(
                    ticker="GME",
                    source="reddit",
                    source_id="synthetic-1",
                    text="$GME moon",
                    url="https://reddit.com/x",
                    created_utc=utcnow(),
                    ingested_at=utcnow(),
                    is_synthetic=True,
                )
            )

        # Seeded store: must refuse rather than silently corrupt.
        with _pytest.raises(SystemExit) as exc:
            assert_store_not_seeded()
        assert exc.value.code == 1

        os.environ.pop("CASHTAG_DATABASE_URL", None)
