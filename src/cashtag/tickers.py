"""Ticker extraction — the hardest correctness problem in this system.

Naive `\\$?[A-Z]{1,5}` matching produces garbage, and the garbage is not evenly
distributed: it concentrates in exactly the tickers whose symbols collide with
English. Three failure modes, each handled below.

1. Real tickers that are also common words.
   "$IT" is Gartner. "$ON" is ON Semiconductor. "$ALL" is Allstate. "$SO" is
   Southern Company. A bare "IT" in "IT department" would score Gartner as the
   most-discussed stock on Reddit, permanently. Handled: bare (unprefixed)
   matches are rejected for any symbol in AMBIGUOUS_BARE; the same symbol is
   accepted when explicitly written "$IT", because the dollar sign is an
   unambiguous statement of intent by the author.

2. Retail-forum slang that looks like a ticker.
   DD, YOLO, ATH, IV, FD, OTM, ITM, PT, ER, EPS. Some of these are also real
   symbols (DD = DuPont, PT, ER). Same mechanism as (1).

3. Symbols appearing inside things that are not prose.
   URLs (reddit.com/r/GME), option contracts, ticker tables, and code blocks all
   contain uppercase runs. URLs and code blocks are stripped before matching;
   option contracts are deliberately KEPT, because "TSLA 250c 7/18" is a genuine
   expression of interest in TSLA.

Every match must also be in the configured universe. An unbounded ticker space
means every typo becomes a stock.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: Explicit cashtag: "$TSLA". The dollar sign is the author telling us they mean
#: a ticker, so this bypasses the ambiguity filter.
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")

#: Bare uppercase symbol: "TSLA". Requires uppercase — "tsla" in running prose is
#: far more often a typo or a word fragment than a ticker, and requiring caps
#: costs us little recall on forums where shouting the ticker is the norm.
BARE_RE = re.compile(r"(?<![$\w])([A-Z]{1,5})(?![\w])")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
CODE_BLOCK_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)

#: Symbols that are real tickers AND common English words / forum slang.
#: A bare occurrence of any of these is rejected; "$SYMBOL" is still accepted.
#: This list is deliberately aggressive. Missing a few GNRC mentions costs recall
#: on one ticker; letting "IT" through corrupts the entire ranking.
AMBIGUOUS_BARE: frozenset[str] = frozenset(
    {
        # Real tickers that are common English words
        "A",
        "ALL",
        "AN",
        "ANY",
        "ARE",
        "AT",
        "BE",
        "BIG",
        "BY",
        "CAN",
        "CAR",
        "CASH",
        "CEO",
        "CFO",
        "CO",
        "COST",
        "DO",
        "EAT",
        "EDIT",
        "EVER",
        "FAST",
        "FOR",
        "FREE",
        "FUN",
        "GO",
        "GOOD",
        "HAS",
        "HE",
        "HOPE",
        "HUGE",
        "IF",
        "IN",
        "IT",
        "JOB",
        "KEY",
        "LAW",
        "LIFE",
        "LOVE",
        "LOW",
        "MAN",
        "ME",
        "MIND",
        "NEW",
        "NEXT",
        "NICE",
        "NO",
        "NOW",
        "OF",
        "OLD",
        "ON",
        "ONE",
        "OPEN",
        "OR",
        "OUT",
        "OWN",
        "PAY",
        "PLAY",
        "POST",
        "REAL",
        "RUN",
        "SAFE",
        "SEE",
        "SO",
        "SUB",
        "TALK",
        "TELL",
        "THE",
        "TO",
        "TRUE",
        "TRY",
        "TWO",
        "UP",
        "US",
        "USA",
        "VERY",
        "WELL",
        "WIN",
        "WORK",
        "YES",
        # Retail-forum and options slang
        "AH",
        "AI",
        "ATH",
        "ATM",
        "BS",
        "BTC",
        "CC",
        "DCA",
        "DD",
        "DTE",
        "EOD",
        "EOW",
        "EPS",
        "ER",
        "ETF",
        "EV",
        "FD",
        "FOMO",
        "FUD",
        "GDP",
        "HODL",
        "IMO",
        "IMHO",
        "IPO",
        "IRA",
        "ITM",
        "IV",
        "LEAP",
        "LEAPS",
        "LMAO",
        "LOL",
        "MOASS",
        "NFA",
        "OMG",
        "OP",
        "OTC",
        "OTM",
        "PE",
        "PM",
        "PT",
        "QE",
        "RH",
        "ROI",
        "RSI",
        "SEC",
        "SL",
        "TA",
        "TL",
        "TLDR",
        "TP",
        "TTM",
        "USD",
        "VIX",
        "WSB",
        "YOLO",
        "YTD",
        # Weekday / month abbreviations that collide with symbols
        "MON",
        "TUE",
        "WED",
        "THU",
        "FRI",
        "SAT",
        "SUN",
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    }
)

#: Fallback universe: liquid, heavily-discussed US equities and ETFs. Deliberately
#: small — precision beats coverage here. Override with a watchlist file.
DEFAULT_UNIVERSE: frozenset[str] = frozenset(
    {
        # Mega-cap tech
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",
        "AMD",
        "INTC",
        "MU",
        "QCOM",
        "ARM",
        "TSM",
        "SMCI",
        "DELL",
        "ORCL",
        "CRM",
        "ADBE",
        "NOW",
        "SNOW",
        "PLTR",
        "CRWD",
        "PANW",
        "ZS",
        "NET",
        "DDOG",
        "MDB",
        "TEAM",
        "SHOP",
        "SQ",
        "PYPL",
        "COIN",
        "HOOD",
        "SOFI",
        "AFRM",
        "UPST",
        # Retail-forum favorites
        "GME",
        "AMC",
        "BB",
        "BBBY",
        "NOK",
        "CLOV",
        "WISH",
        "SNDL",
        "MULN",
        "RIVN",
        "LCID",
        "NIO",
        "XPEV",
        "LI",
        "FSR",
        "NKLA",
        "SPCE",
        "DKNG",
        "PENN",
        "RBLX",
        "U",
        "DWAC",
        "PHUN",
        "MMTLP",
        "ATER",
        "PROG",
        "BBIG",
        "SPRT",
        "IRNT",
        # Broad-market and sector ETFs
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VOO",
        "VTI",
        "ARKK",
        "XLF",
        "XLE",
        "XLK",
        "XLV",
        "XLI",
        "XLU",
        "XLP",
        "XLY",
        "XLB",
        "XLRE",
        "SMH",
        "SOXX",
        "TQQQ",
        "SQQQ",
        "UVXY",
        "VXX",
        "SPXU",
        "SPXL",
        "TLT",
        "HYG",
        "GLD",
        "SLV",
        "USO",
        # Financials, healthcare, industrials, energy, consumer
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "SCHW",
        "BLK",
        "AXP",
        "V",
        "MA",
        "BRK.B",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "TMO",
        "ABT",
        "CVS",
        "MRNA",
        "BNTX",
        "NVAX",
        "BA",
        "CAT",
        "DE",
        "GE",
        "HON",
        "LMT",
        "RTX",
        "NOC",
        "GD",
        "UPS",
        "FDX",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "OXY",
        "PSX",
        "WMT",
        "TGT",
        "HD",
        "LOW",
        "MCD",
        "SBUX",
        "NKE",
        "DIS",
        "KO",
        "PEP",
        "PG",
        "CL",
        "F",
        "GM",
        "UBER",
        "LYFT",
        "ABNB",
        "DASH",
        "CVNA",
        "CHWY",
        "ETSY",
        "EBAY",
        "BABA",
        "JD",
        "PDD",
        "SE",
        "MELI",
        "T",
        "VZ",
        "TMUS",
    }
)

_WATCHLIST_PATH = Path(__file__).parent / "data" / "universe.txt"


@lru_cache(maxsize=1)
def load_universe() -> frozenset[str]:
    """Load the tracked ticker universe.

    Reads `data/universe.txt` (one symbol per line, '#' comments allowed) if it
    exists, else falls back to DEFAULT_UNIVERSE. Point this at an export from
    your own screener to track a different book.
    """
    if _WATCHLIST_PATH.exists():
        symbols = set()
        for raw in _WATCHLIST_PATH.read_text().splitlines():
            line = raw.split("#", 1)[0].strip().upper()
            if line:
                symbols.add(line.lstrip("$"))
        if symbols:
            return frozenset(symbols)
    return DEFAULT_UNIVERSE


def _strip_noise(text: str) -> str:
    """Remove URLs and code blocks, which contain uppercase runs that are not prose.

    'reddit.com/r/GME' is a link, not someone expressing a view on GameStop.
    """
    text = CODE_BLOCK_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return text


def extract_tickers(text: str, universe: frozenset[str] | None = None) -> set[str]:
    """Extract ticker symbols mentioned in `text`.

    Args:
        text: Raw post title + body, or caption.
        universe: Tracked symbols. Defaults to `load_universe()`.

    Returns:
        Uppercase symbols, no '$' prefix. Empty set when nothing matches.

    Rules:
        - "$XYZ" is accepted whenever XYZ is in the universe, even if XYZ is an
          ambiguous word — the author's '$' is an explicit disambiguation.
        - Bare "XYZ" is accepted only if XYZ is in the universe AND not in
          AMBIGUOUS_BARE.
        - URLs and code blocks are excluded from matching entirely.
    """
    if not text:
        return set()
    universe = universe if universe is not None else load_universe()
    cleaned = _strip_noise(text)

    found: set[str] = set()

    # Explicit cashtags: trusted, ambiguity filter does not apply.
    for match in CASHTAG_RE.finditer(cleaned):
        symbol = match.group(1).upper()
        if symbol in universe:
            found.add(symbol)

    # Bare symbols: must clear the ambiguity filter.
    for match in BARE_RE.finditer(cleaned):
        symbol = match.group(1)
        if symbol in universe and symbol not in AMBIGUOUS_BARE:
            found.add(symbol)

    return found


def extract_with_confidence(text: str, universe: frozenset[str] | None = None) -> dict[str, str]:
    """Like `extract_tickers`, but reports HOW each symbol matched.

    Returns a mapping of symbol -> "cashtag" (explicit '$', high confidence) or
    "bare" (unprefixed, lower confidence). Not used for scoring today; it exists
    so a future weighting scheme can discount bare matches without re-parsing,
    and so extraction decisions are auditable when a result looks wrong.
    """
    if not text:
        return {}
    universe = universe if universe is not None else load_universe()
    cleaned = _strip_noise(text)

    result: dict[str, str] = {}
    for match in BARE_RE.finditer(cleaned):
        symbol = match.group(1)
        if symbol in universe and symbol not in AMBIGUOUS_BARE:
            result[symbol] = "bare"
    # Cashtags win: a symbol appearing both ways is recorded at the higher confidence.
    for match in CASHTAG_RE.finditer(cleaned):
        symbol = match.group(1).upper()
        if symbol in universe:
            result[symbol] = "cashtag"
    return result
