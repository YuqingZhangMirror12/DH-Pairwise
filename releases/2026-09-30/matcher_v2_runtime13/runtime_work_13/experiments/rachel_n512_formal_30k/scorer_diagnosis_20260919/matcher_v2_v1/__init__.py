"""Opt-in Matcher v2; no import-time mutation of legacy experiments."""

from .adapter import MatcherV2Adapter, fresh_matcher_v2
from .network import MatcherV2Config

__all__ = ['MatcherV2Adapter', 'MatcherV2Config', 'fresh_matcher_v2']
