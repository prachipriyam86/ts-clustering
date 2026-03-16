"""
batch_dispatcher.py
-------------------
Converts clusters into reviewer-ready batches with context frames.

Each batch includes:
  - The cluster's context frame (vertical, flag type, confidence, count)
  - The ordered list of item IDs
  - Quality metadata for monitoring

In a real system, dispatch_batch() would push to a review queue API.
Here it serialises to a dict for demonstration.
"""

import json
from dataclasses import asdict, dataclass
from typing import List, Dict, Any

from clusterer import Cluster


@dataclass
class ReviewBatch:
    batch_id: str
    context_frame: str
    vertical: str
    dominant_flag: str
    avg_confidence: float
    size: int
    item_ids: List[str]


class BatchDispatcher:
    """
    Converts clusters into structured reviewer batches.
    """

    def build_batch(self, cluster: Cluster) -> ReviewBatch:
        return ReviewBatch(
            batch_id=cluster.cluster_id,
            context_frame=cluster.context_frame,
            vertical=cluster.vertical,
            dominant_flag=cluster.dominant_flag,
            avg_confidence=round(cluster.avg_confidence, 3),
            size=len(cluster.items),
            item_ids=[item.item_id for item in cluster.items],
        )

    def dispatch_all(self, clusters: List[Cluster]) -> List[ReviewBatch]:
        """Build and return all reviewer batches."""
        return [self.build_batch(c) for c in clusters]

    def to_json(self, batches: List[ReviewBatch], path: str) -> None:
        """Serialise batches to a JSON file."""
        payload = [asdict(b) for b in batches]
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Dispatched {len(batches)} batches to {path}")
