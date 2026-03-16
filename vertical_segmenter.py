"""
vertical_segmenter.py
---------------------
Layer 1 of the two-layer clustering architecture.

Classifies incoming flagged content items into verticals
(gaming, children's, music, news, cooking, etc.) before
any clustering occurs.

In production, vertical classification typically comes from
upstream metadata or a fast lightweight classifier. This module
provides both a metadata-based path and a simple rule-based
fallback for demonstration purposes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


VERTICALS = [
    "gaming",
    "childrens",
    "music",
    "news",
    "cooking",
    "sports",
    "education",
    "other",
]


@dataclass
class ContentItem:
    """Represents a single flagged content item in the review queue."""
    item_id: str
    vertical: Optional[str]          # from upstream metadata if available
    flag_type: str                    # violence / spam / copyright / misleading / safe
    flag_confidence: float            # classifier confidence [0-1]
    audio_signal: float               # audio modality score [0-1]
    visual_signal: float              # visual modality score [0-1]
    text_signal: float                # text/metadata signal score [0-1]
    channel_violation_rate: float     # historical channel violation rate [0-1]
    account_type: str                 # verified / established / new / anonymous
    reviewer_decision: Optional[str] = None  # remove / allow / escalate (ground truth)


class VerticalSegmenter:
    """
    Segments incoming content items into vertical buckets.

    Priority order:
    1. Use vertical from upstream metadata if present and valid
    2. Fall back to rule-based classification on flag_type + account signals

    In a real system, step 2 would be a trained lightweight classifier.
    """

    def __init__(self, valid_verticals: List[str] = VERTICALS):
        self.valid_verticals = set(valid_verticals)

    def classify(self, item: ContentItem) -> str:
        """Return the vertical for a single item."""
        # Use upstream metadata if available and valid
        if item.vertical and item.vertical in self.valid_verticals:
            return item.vertical
        # Fallback: rule-based on flag signals
        return self._rule_based_fallback(item)

    def _rule_based_fallback(self, item: ContentItem) -> str:
        """
        Simple heuristic fallback.
        Replace with a trained classifier in production.
        """
        if item.flag_type == "copyright":
            return "music"
        if item.account_type == "verified" and item.text_signal > 0.7:
            return "news"
        return "other"

    def segment(self, items: List[ContentItem]) -> Dict[str, List[ContentItem]]:
        """
        Segment a list of items into vertical buckets.

        Returns:
            Dict mapping vertical name -> list of ContentItems
        """
        buckets: Dict[str, List[ContentItem]] = {v: [] for v in self.valid_verticals}

        for item in items:
            vertical = self.classify(item)
            if vertical not in buckets:
                buckets[vertical] = []
            buckets[vertical].append(item)

        # Remove empty buckets
        return {v: items for v, items in buckets.items() if items}
