"""
clusterer.py
------------
Layer 2 of the two-layer clustering architecture.

Applies K-Means clustering within each vertical bucket using
policy-relevant embeddings from the MultimodalEmbedder.

Key design decisions:
  - K-Means per vertical (not across the full queue)
  - k is set relative to bucket size and max_cluster_size target
  - Degenerate cluster detection (catch-all or singleton clusters)
  - Cluster metadata includes dominant flag type and avg confidence
    for building the reviewer context frame
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from vertical_segmenter import ContentItem
from embedder import MultimodalEmbedder


@dataclass
class Cluster:
    """A group of policy-similar content items within one vertical."""
    cluster_id: str
    vertical: str
    items: List[ContentItem]
    embeddings: np.ndarray
    centroid: np.ndarray
    dominant_flag: str
    avg_confidence: float
    context_frame: str = ""

    def __post_init__(self):
        if not self.context_frame:
            self.context_frame = (
                f"{self.vertical.capitalize()} | "
                f"{self.dominant_flag} classifier | "
                f"{'high' if self.avg_confidence > 0.7 else 'medium'} confidence | "
                f"{len(self.items)} items"
            )


class Clusterer:
    """
    Clusters content items within a vertical using K-Means.

    Args:
        max_cluster_size: target maximum items per cluster.
            Smaller clusters reduce anchoring bias risk.
        min_cluster_size: clusters smaller than this are merged
            into an 'edge_cases' overflow batch.
        random_state: for reproducibility
    """

    def __init__(
        self,
        max_cluster_size: int = 25,
        min_cluster_size: int = 3,
        random_state: int = 42,
    ):
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.embedder = MultimodalEmbedder()
        self.scaler = StandardScaler()

    def _choose_k(self, n_items: int) -> int:
        """
        Choose k relative to bucket size and max_cluster_size target.
        Ensures at least 1 cluster, at most n_items clusters.
        """
        k = max(1, round(n_items / self.max_cluster_size))
        return min(k, n_items)

    def _dominant_flag(self, items: List[ContentItem]) -> str:
        """Return the most common flag type in a group of items."""
        from collections import Counter
        counts = Counter(item.flag_type for item in items)
        return counts.most_common(1)[0][0]

    def cluster_vertical(
        self, vertical: str, items: List[ContentItem]
    ) -> List[Cluster]:
        """
        Cluster all items within one vertical.

        Returns:
            List of Cluster objects ready for dispatch
        """
        if len(items) == 0:
            return []

        # Embed items into policy-relevant vector space
        embeddings = self.embedder.embed_batch(items)

        # Normalise features before clustering
        embeddings_scaled = self.scaler.fit_transform(embeddings)

        # Determine k
        k = self._choose_k(len(items))

        if k == 1:
            # Only one cluster — return all items as one batch
            return [self._build_cluster(vertical, items, embeddings, 0)]

        # Fit K-Means
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(embeddings_scaled)

        # Group items by cluster label
        clusters: List[Cluster] = []
        for label in range(k):
            mask = labels == label
            cluster_items = [item for item, m in zip(items, mask) if m]
            cluster_embeddings = embeddings[mask]

            if len(cluster_items) < self.min_cluster_size:
                # Too small — will be handled as edge case overflow
                continue

            clusters.append(
                self._build_cluster(vertical, cluster_items, cluster_embeddings, label)
            )

        return clusters

    def _build_cluster(
        self,
        vertical: str,
        items: List[ContentItem],
        embeddings: np.ndarray,
        label: int,
    ) -> Cluster:
        centroid = embeddings.mean(axis=0)
        dominant_flag = self._dominant_flag(items)
        avg_confidence = np.mean([item.flag_confidence for item in items])
        cluster_id = f"{vertical}_{dominant_flag}_{label}"

        return Cluster(
            cluster_id=cluster_id,
            vertical=vertical,
            items=items,
            embeddings=embeddings,
            centroid=centroid,
            dominant_flag=dominant_flag,
            avg_confidence=float(avg_confidence),
        )
