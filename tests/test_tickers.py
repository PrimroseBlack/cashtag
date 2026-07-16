"""Tests for ticker extraction.

Precision is what matters here, and the tests are weighted accordingly. A false
positive on an ambiguous symbol does not cost one bad row — it corrupts the
ranking permanently, because "IT" appears in a meaningful fraction of all English
prose ever written. Recall misses cost one ticker some mentions.
"""

from __future__ import annotations

from cashtag.tickers import extract_tickers, extract_with_confidence

UNIVERSE = frozenset({"TSLA", "GME", "AMC", "NVDA", "IT", "ON", "ALL", "SO", "DD", "PT", "A", "F"})


class TestExplicitCashtags:
    def test_extracts_a_simple_cashtag(self):
        assert extract_tickers("$TSLA is going up", UNIVERSE) == {"TSLA"}

    def test_extracts_multiple_cashtags(self):
        assert extract_tickers("$GME and $AMC squeeze", UNIVERSE) == {"GME", "AMC"}

    def test_cashtag_is_case_insensitive(self):
        assert extract_tickers("$tsla calls", UNIVERSE) == {"TSLA"}

    def test_cashtag_overrides_the_ambiguity_filter(self):
        # The author wrote '$IT'. The dollar sign is them saying "I mean Gartner".
        assert extract_tickers("$IT had good earnings", UNIVERSE) == {"IT"}

    def test_cashtag_for_ambiguous_word_on_is_accepted(self):
        assert extract_tickers("$ON semiconductor is undervalued", UNIVERSE) == {"ON"}

    def test_single_letter_cashtag(self):
        assert extract_tickers("$F is a value play", UNIVERSE) == {"F"}


class TestBareSymbols:
    def test_extracts_unambiguous_bare_symbol(self):
        assert extract_tickers("TSLA is going up", UNIVERSE) == {"TSLA"}

    def test_requires_uppercase(self):
        # Lowercase in running prose is far more often a word than a ticker.
        assert extract_tickers("tsla is going up", UNIVERSE) == set()

    def test_rejects_symbols_outside_the_universe(self):
        # Unbounded ticker space means every typo becomes a stock.
        assert extract_tickers("ZZZZ to the moon", UNIVERSE) == set()

    def test_rejects_bare_cashtag_lookalike_in_word(self):
        assert extract_tickers("TSLAQ is a movement", UNIVERSE) == set()


class TestAmbiguityFilter:
    """The reason the project is named Cashtag."""

    def test_bare_it_is_not_gartner(self):
        # Without this filter, Gartner is permanently the most-discussed stock
        # on Reddit — "IT" appears in an enormous share of English prose.
        assert extract_tickers("I work in IT and I like stocks", UNIVERSE) == set()

    def test_bare_on_is_not_on_semiconductor(self):
        assert extract_tickers("PUT ON YOUR SEATBELT", UNIVERSE) == set()

    def test_bare_all_is_not_allstate(self):
        assert extract_tickers("ALL IN ON THIS TRADE", UNIVERSE) == set()

    def test_bare_so_is_not_southern_company(self):
        assert extract_tickers("SO I bought some calls", UNIVERSE) == set()

    def test_bare_dd_is_slang_not_dupont(self):
        # "DD" on these forums means due diligence approximately always.
        assert extract_tickers("Here is my DD on this company", UNIVERSE) == set()

    def test_bare_pt_is_price_target_not_the_ticker(self):
        assert extract_tickers("My PT is 400 by year end", UNIVERSE) == set()

    def test_ambiguous_words_do_not_suppress_valid_neighbors(self):
        # The filter must reject the ambiguous token only, not the whole post.
        assert extract_tickers("ALL IN ON $GME", UNIVERSE) == {"GME"}


class TestNoiseStripping:
    def test_ignores_symbols_inside_urls(self):
        # A link to a subreddit is not an opinion about the stock.
        text = "See https://reddit.com/r/GME for more"
        assert extract_tickers(text, UNIVERSE) == set()

    def test_ignores_symbols_inside_code_blocks(self):
        text = "```\nticker = TSLA\n```"
        assert extract_tickers(text, UNIVERSE) == set()

    def test_ignores_symbols_in_inline_code(self):
        assert extract_tickers("Use `NVDA` as the arg", UNIVERSE) == set()

    def test_extracts_prose_around_a_stripped_url(self):
        text = "$TSLA news here https://reddit.com/r/GME"
        assert extract_tickers(text, UNIVERSE) == {"TSLA"}


class TestOptionContracts:
    def test_option_contract_still_counts_as_a_mention(self):
        # Deliberately KEPT. "TSLA 250c 7/18" is a real expression of interest.
        assert extract_tickers("TSLA 250c 7/18 printing", UNIVERSE) == {"TSLA"}

    def test_zero_dte_mention_counts(self):
        assert extract_tickers("$NVDA 0DTE gamble", UNIVERSE) == {"NVDA"}


class TestEdgeCases:
    def test_empty_string(self):
        assert extract_tickers("", UNIVERSE) == set()

    def test_whitespace_only(self):
        assert extract_tickers("   \n  ", UNIVERSE) == set()

    def test_bare_dollar_sign(self):
        assert extract_tickers("I lost $ today", UNIVERSE) == set()

    def test_dollar_amount_is_not_a_ticker(self):
        assert extract_tickers("I made $5000 today", UNIVERSE) == set()

    def test_deduplicates_repeated_mentions(self):
        assert extract_tickers("$GME $GME GME to the moon", UNIVERSE) == {"GME"}


class TestConfidence:
    def test_cashtag_match_is_reported_as_high_confidence(self):
        assert extract_with_confidence("$TSLA up", UNIVERSE) == {"TSLA": "cashtag"}

    def test_bare_match_is_reported_as_lower_confidence(self):
        assert extract_with_confidence("TSLA up", UNIVERSE) == {"TSLA": "bare"}

    def test_cashtag_wins_when_a_symbol_appears_both_ways(self):
        assert extract_with_confidence("TSLA up, $TSLA calls", UNIVERSE) == {"TSLA": "cashtag"}
