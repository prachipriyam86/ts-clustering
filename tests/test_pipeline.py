"""
test_pipeline.py
----------------
Unit tests for the T&S clustering pipeline.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from vertical_segmenter import ContentItem, VerticalSegmenter
from embedder import MultimodalEmbedder
from clusterer import Clusterer
from quality_evaluator import QualityEvaluator, AGREEMENT_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_item(item_id="i1", vertical="gaming", flag_type="violence",
              flag_confidence=0.85, audio_signal=0.6, visual_signal=0.8,
              text_signal=0.3, channel_violation_rate=0.2,
              account_type="established", reviewer_decision=None):
    return ContentItem(
        item_id=item_id, vertical=vertical, flag_type=flag_type,
        flag_confidence=flag_confidence, audio_signal=audio_signal,
        visual_signal=visual_signal, text_signal=text_signal,
        channel_violation_rate=channel_violation_rate,
        account_type=account_type, reviewer_decision=reviewer_decision,
    )


def make_items(n, vertical="gaming", flag_type="violence", decision="remove"):
    return [
        make_item(item_id=f"i{i}", vertical=vertical,
                  flag_type=flag_type, reviewer_decision=decision)
        for i in range(n)
    ]


# ── VerticalSegmenter ─────────────────────────────────────────────────────────

class TestVerticalSegmenter:

    def test_uses_metadata_vertical(self):
        segmenter = VerticalSegmenter()
        item = make_item(vertical="gaming")
        assert segmenter.classify(item) == "gaming"

    def test_fallback_copyright_to_music(self):
        segmenter = VerticalSegmenter()
        item = make_item(vertical=None, flag_type="copyright")
        assert segmenter.classify(item) == "music"

    def test_segment_groups_by_vertical(self):
        segmenter = VerticalSegmenter()
        items = [make_item(item_id="a", vertical="gaming"),
                 make_item(item_id="b", vertical="music"),
                 make_item(item_id="c", vertical="gaming")]
        buckets = segmenter.segment(items)
        assert len(buckets["gaming"]) == 2
        assert len(buckets["music"]) == 1

    def test_empty_queue_returns_empty_buckets(self):
        segmenter = VerticalSegmenter()
        buckets = segmenter.segment([])
        assert all(len(v) == 0 for v in buckets.values())


# ── MultimodalEmbedder ────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_returns_vector(self):
        embedder = MultimodalEmbedder()
        item = make_item()
        vec = embedder.embed(item)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 7

    def test_embed_batch_shape(self):
        embedder = MultimodalEmbedder()
        items = make_items(10)
        matrix = embedder.embed_batch(items)
        assert matrix.shape == (10, 7)

    def test_different_items_produce_different_embeddings(self):
        embedder = MultimodalEmbedder()
        item_a = make_item(flag_confidence=0.9, visual_signal=0.9)
        item_b = make_item(flag_confidence=0.1, visual_signal=0.1)
        assert not np.allclose(embedder.embed(item_a), embedder.embed(item_b))


# ── Clusterer ─────────────────────────────────────────────────────────────────

class TestClusterer:

    def test_cluster_returns_list(self):
        clusterer = Clusterer(max_cluster_size=10)
        items = make_items(20)
        clusters = clusterer.cluster_vertical("gaming", items)
        assert isinstance(clusters, list)
        assert len(clusters) >= 1

    def test_total_items_preserved(self):
        clusterer = Clusterer(max_cluster_size=10, min_cluster_size=1)
        items = make_items(30)
        clusters = clusterer.cluster_vertical("gaming", items)
        total = sum(len(c.items) for c in clusters)
        assert total == 30

    def test_empty_vertical_returns_empty(self):
        clusterer = Clusterer()
        assert clusterer.cluster_vertical("gaming", []) == []

    def test_cluster_ids_contain_vertical(self):
        clusterer = Clusterer(max_cluster_size=50)
        items = make_items(5)
        clusters = clusterer.cluster_vertical("gaming", items)
        for c in clusters:
            assert "gaming" in c.cluster_id

    def test_single_item_returns_one_cluster(self):
        clusterer = Clusterer(max_cluster_size=25, min_cluster_size=1)
        items = make_items(1)
        clusters = clusterer.cluster_vertical("gaming", items)
        assert len(clusters) == 1


# ── QualityEvaluator ──────────────────────────────────────────────────────────

class TestQualityEvaluator:

    def _make_cluster_with_decisions(self, decisions):
        from clusterer import Cluster
        items = [
            make_item(item_id=f"i{i}", reviewer_decision=d)
            for i, d in enumerate(decisions)
        ]
        embedder = MultimodalEmbedder()
        embeddings = embedder.embed_batch(items)
        return Cluster(
            cluster_id="gaming_violence_0",
            vertical="gaming",
            items=items,
            embeddings=embeddings,
            centroid=embeddings.mean(axis=0),
            dominant_flag="violence",
            avg_confidence=0.85,
        )

    def test_high_agreement_is_good(self):
        evaluator = QualityEvaluator()
        cluster = self._make_cluster_with_decisions(
            ["remove"] * 9 + ["allow"]  # 90% agreement
        )
        report = evaluator.evaluate(cluster)
        assert report.quality_status == "good"

    def test_no_decisions_is_pending(self):
        evaluator = QualityEvaluator()
        cluster = self._make_cluster_with_decisions([None] * 5)
        report = evaluator.evaluate(cluster)
        assert report.quality_status == "pending"

    def test_low_agreement_triggers_flag(self):
        evaluator = QualityEvaluator(agreement_threshold=0.75)
        cluster = self._make_cluster_with_decisions(
            ["remove", "allow", "remove", "escalate",
             "allow", "remove", "allow", "escalate"]  # ~37% majority
        )
        report = evaluator.evaluate(cluster)
        assert report.quality_status in ("policy_gap", "tune_parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
