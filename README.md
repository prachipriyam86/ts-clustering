# Trust & Safety Review Queue Clustering

This repo demonstrates how to apply K-Means clustering with multimodal (text + metadata) embeddings to structure content moderation review queues — reducing context-switching overhead and improving reviewer agreement rates.

---

## The Problem

Most Trust & Safety review queues dispatch content in FIFO or random order. Reviewers encounter a gaming clip, then a children's tutorial, then a music video — each requiring a full mental reset to a different policy framework. This context switching is both slow and a source of inconsistency.

**This architecture fixes that by grouping policy-similar content into batches before it reaches reviewers.**

---

## Architecture Overview

```
Incoming Flagged Queue
        |
        v
Layer 1: Vertical Segmentation
(gaming / children's / music / news / ...)
        |
        v
Layer 2: K-Means Clustering Within Each Vertical
(using multimodal policy-relevant embeddings)
        |
        v
Reviewer Batches
[Vertical | Flag Type | Confidence | N items]
```

---

## Repository Structure

'''
ts-clustering/
├── src/
│   ├── vertical_segmenter.py      # Layer 1: classify content by vertical
│   ├── embedder.py                # Multimodal feature fusion
│   ├── clusterer.py               # K-Means clustering per vertical
│   ├── batch_dispatcher.py        # Build reviewer-ready batches
│   ├── quality_evaluator.py       # Cluster quality: agreement rate + silhouette
│   └── pipeline.py                # End-to-end pipeline
├── notebooks/
│   └── demo.ipynb                 # Walkthrough with sample data
├── tests/
│   └── test_pipeline.py           # Unit tests
├── data/
│   └── sample/
│       └── sample_queue.csv       # Synthetic sample data
├── requirements.txt
└── README.md
'''

---

## Quick Start

'''bash
git clone https://github.com/prachipriyam86/ts-clustering
cd ts-clustering
pip install -r requirements.txt

# Run the full pipeline on sample data
python src/pipeline.py --input data/sample/sample_queue.csv --output batches_out.json
'''

---

## Sample Data

The included 'sample_queue.csv' is **synthetic** — generated to demonstrate the pipeline structure. It contains no real user content or proprietary platform data.

| Field | Description |
|---|---|
| 'item_id' | Unique item identifier |
| 'vertical' | Content vertical (gaming, childrens, music, news, cooking) |
| 'flag_type' | Classifier flag (violence, spam, copyright, misleading, safe) |
| 'flag_confidence' | Classifier confidence score [0–1] |
| 'audio_signal' | Audio modality score [0–1] |
| 'visual_signal' | Visual modality score [0–1] |
| 'text_signal' | Text/metadata signal score [0–1] |
| 'channel_violation_rate' | Historical channel violation rate [0–1] |
| 'account_type' | verified / established / new / anonymous |
| 'reviewer_decision' | Ground truth label (remove / allow / escalate) |

---

## Key Design Decisions

**Why K-Means over HDBSCAN?**  
At production queue volumes, K-Means offers better latency and interpretability. HDBSCAN handles variable-density clusters more naturally but is harder to tune for real-time dispatch. K-Means per vertical with monitored k sidesteps most of the equal-cluster-size limitation.

**Why reviewer agreement rate as the quality metric?**  
Silhouette score measures geometric cluster coherence. Reviewer agreement rate measures whether the cluster is actually useful for the review task. A cluster that is geometrically tight but generates reviewer disagreement is a bad cluster for this application — it signals a policy gap, not a clustering win.

**Why multimodal embeddings?**  
Visual similarity ≠ policy similarity. A graphic medical procedure on a verified healthcare channel is not the same review task as identical content from an anonymous account. The embedding layer fuses visual/audio signals with context signals to produce a policy-relevant vector space.

---

## Cluster Quality Output

The pipeline outputs per-cluster quality metrics:

'''json
{
  "cluster_id": "gaming_violence_3",
  "vertical": "gaming",
  "dominant_flag": "violence",
  "size": 18,
  "silhouette_score": 0.61,
  "reviewer_agreement_rate": 0.89,
  "quality_status": "good",
  "context_frame": "Gaming | violence classifier | high confidence | 18 items"
}
'''

---

## Citation

If you use this code in your work, please cite:

```
Priyam, P. (2026). Trust & Safety Review Queue Clustering.
GitHub. https://github.com/prachipriyam86/ts-clustering
```

---

## License

MIT License. See 'LICENSE' for details.
