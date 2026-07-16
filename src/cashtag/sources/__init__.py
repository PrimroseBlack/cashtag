"""Ingestion sources. Each uses a different access method — see README "Four sources, four access models"."""

from .base import IngestionSource, RawPost
from .brightdata import InstagramSource, TikTokSource
from .reddit import RedditSource
from .stocktwits import StockTwitsSource
from .xapi import XSource

__all__ = [
    "IngestionSource",
    "RawPost",
    "RedditSource",
    "XSource",
    "InstagramSource",
    "TikTokSource",
    "StockTwitsSource",
]
