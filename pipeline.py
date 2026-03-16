"""
pipeline.py
-----------
End-to-end pipeline: flagged queue -> reviewer batches.

Usage:
    python src/pipeline.py --input data/sample/sample_queue.csv \
                           --output batches_out.json \
                           --quality_report quality_report.json
"""

import argparse
import csv
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vertical_segmenter import ContentItem, VerticalSegmenter
from clusterer import Clusterer
from batch_dispatcher import BatchDispatcher
from quality_evaluator import QualityEvaluator


def load_queue(csv_path: str):
    """Load flagged content items from a CSV file."""
    items = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(ContentItem(
                item_id=row["item_id"],
                vertical=row.get("vertical") or None,
                flag_type=row["flag_type"],
                flag_confidence=float(row["flag_confidence"]),
                audio_signal=float(row["audio_signal"]),
                visual_signal=float(row["visual_signal"]),
                text_signal=float(row["text_signal"]),
                channel_violation_rate=float(row["channel_violation_rate"]),
                account_type=row["account_type"],
                reviewer_decision=row.get("reviewer_decision") or None,
            ))
    return items


def run_pipeline(
    input_path: str,
    output_path: str,
    quality_report_path: str = None,
    max_cluster_size: int = 25,
):
    print(f"\n=== Trust & Safety Clustering Pipeline ===\n")

    # Load
    items = load_queue(input_path)
    print(f"Loaded {len(items)} items from queue.")

    # Layer 1: Vertical segmentation
    segmenter = VerticalSegmenter()
    buckets = segmenter.segment(items)
    print(f"\nLayer 1 — Vertical segmentation:")
    for vertical, bucket_items in buckets.items():
        print(f"  {vertical}: {len(bucket_items)} items")

    # Layer 2: Clustering within each vertical
    clusterer = Clusterer(max_cluster_size=max_cluster_size)
    all_clusters = []
    print(f"\nLayer 2 — K-Means clustering (max_cluster_size={max_cluster_size}):")
    for vertical, bucket_items in buckets.items():
        clusters = clusterer.cluster_vertical(vertical, bucket_items)
        all_clusters.extend(clusters)
        print(f"  {vertical}: {len(bucket_items)} items -> {len(clusters)} clusters")

    # Dispatch batches
    dispatcher = BatchDispatcher()
    batches = dispatcher.dispatch_all(all_clusters)
    dispatcher.to_json(batches, output_path)

    print(f"\nReviewer batches:")
    for batch in batches:
        print(f"  [{batch.batch_id}]  {batch.context_frame}")

    # Quality evaluation (if reviewer decisions are present)
    evaluator = QualityEvaluator()
    reports = evaluator.evaluate_all(all_clusters)

    pending = [r for r in reports if r.quality_status == "pending"]
    if len(pending) == len(reports):
        print(f"\nQuality evaluation: no reviewer decisions available yet.")
    else:
        policy_gaps = [r for r in reports if r.quality_status == "policy_gap"]
        good = [r for r in reports if r.quality_status == "good"]
        print(f"\nQuality evaluation:")
        print(f"  Good clusters:   {len(good)}")
        print(f"  Policy gaps:     {len(policy_gaps)}")
        if policy_gaps:
            print(f"\n  Policy gap clusters (flag for policy team):")
            for r in policy_gaps:
                print(f"    - {r.cluster_id}: agreement={r.reviewer_agreement_rate:.2f}")
                print(f"      {r.action}")

    if quality_report_path:
        from dataclasses import asdict
        with open(quality_report_path, "w") as f:
            json.dump([asdict(r) for r in reports], f, indent=2)
        print(f"\nQuality report saved to {quality_report_path}")

    print(f"\nDone. {len(batches)} batches dispatched.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T&S Review Queue Clustering Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV queue file")
    parser.add_argument("--output", required=True, help="Path for output batches JSON")
    parser.add_argument("--quality_report", default=None, help="Path for quality report JSON")
    parser.add_argument("--max_cluster_size", type=int, default=25)
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        quality_report_path=args.quality_report,
        max_cluster_size=args.max_cluster_size,
    )
