"""
quality_evaluator.py
--------------------
Evaluates cluster quality using both geometric and operational metrics.

The primary signal is reviewer_agreement_rate — whether reviewers
actually agree on items within the cluster. This is the metric that
reflects real-world utility for the review task.

Silhouette score is computed as a secondary diagnostic signal.

Quality statuses:
  - good:              high agreement, reasonably tight geometry
  - policy_gap:        low agreement on a tight cluster
                       --> policy guidelines are ambiguous for this type
  - needs_retraining:  low agreement on a loose cluster
                       --> embedding space needs improvement
  - tune_parameters:   low agreement, loose geometry, small cluster
                       --> adjust k or max_cluster_size
"""

from dataclasses import dataclass
from typing import List, Optional
from collections import Counter

import numpy as np
from sklearn.metrics import silhouette_score

from clusterer import Cluster


AGREEMENT_THRESHOLD = 0.75   # below this triggers quality flag
SILHOUETTE_THRESHOLD = 0.40  # below this = geometrically loose


@dataclass
class ClusterQualityReport:
    cluster_id: str
    vertical: str
    dominant_flag: str
    size: int
    silhouette_score: Optional[float]
    reviewer_agreement_rate: Optional[float]
    quality_status: str          # good / policy_gap / needs_retraining / tune_parameters
    context_frame: str
    action: str                  # human-readable recommendation


class QualityEvaluator:
    """
    Evaluates cluster quality after reviewer decisions are recorded.

    Usage:
        evaluator = QualityEvaluator()
        report = evaluator.evaluate(cluster)
    """

    def __init__(
        self,
        agreement_threshold: float = AGREEMENT_THRESHOLD,
        silhouette_threshold: float = SILHOUETTE_THRESHOLD,
    ):
        self.agreement_threshold = agreement_threshold
        self.silhouette_threshold = silhouette_threshold

    def _reviewer_agreement_rate(self, cluster: Cluster) -> Optional[float]:
        """
        Compute the fraction of items where the reviewer decision
        matches the majority decision in the cluster.

        Returns None if reviewer decisions are not yet available.
        """
        decisions = [
            item.reviewer_decision
            for item in cluster.items
            if item.reviewer_decision is not None
        ]
        if len(decisions) < 2:
            return None

        majority = Counter(decisions).most_common(1)[0][0]
        agreement = sum(1 for d in decisions if d == majority) / len(decisions)
        return agreement

    def _silhouette(self, cluster: Cluster) -> Optional[float]:
        """
        Silhouette score requires at least 2 items.
        Returns None for trivially small clusters.
        """
        if len(cluster.items) < 2:
            return None
        try:
            # Silhouette needs at least 2 distinct labels — use within-cluster
            # distances as a proxy: compute pairwise distances to centroid
            dists = np.linalg.norm(
                cluster.embeddings - cluster.centroid, axis=1
            )
            # Normalise to [0, 1] as a cohesion proxy (lower = tighter)
            cohesion = float(1.0 - np.mean(dists) / (np.max(dists) + 1e-9))
            return round(cohesion, 3)
        except Exception:
            return None

    def evaluate(self, cluster: Cluster) -> ClusterQualityReport:
        """Evaluate a single cluster and return a quality report."""

        sil = self._silhouette(cluster)
        agreement = self._reviewer_agreement_rate(cluster)

        # Determine status and action
        if agreement is None:
            status = "pending"
            action = "Awaiting reviewer decisions."
        elif agreement >= self.agreement_threshold:
            status = "good"
            action = "No action needed."
        else:
            # Low agreement — diagnose via silhouette
            if sil is not None and sil >= self.silhouette_threshold:
                status = "policy_gap"
                action = (
                    "Cluster is geometrically tight but reviewers disagree. "
                    "Flag for policy team: guidelines may be ambiguous for "
                    f"'{cluster.dominant_flag}' content in '{cluster.vertical}' vertical."
                )
            else:
                status = "tune_parameters"
                action = (
                    "Loose cluster with low agreement. "
                    "Consider reducing max_cluster_size or retraining embeddings "
                    f"for '{cluster.vertical}' vertical."
                )

        return ClusterQualityReport(
            cluster_id=cluster.cluster_id,
            vertical=cluster.vertical,
            dominant_flag=cluster.dominant_flag,
            size=len(cluster.items),
            silhouette_score=sil,
            reviewer_agreement_rate=agreement,
            quality_status=status,
            context_frame=cluster.context_frame,
            action=action,
        )

    def evaluate_all(self, clusters: List[Cluster]) -> List[ClusterQualityReport]:
        """Evaluate a list of clusters and return all reports."""
        return [self.evaluate(c) for c in clusters]
