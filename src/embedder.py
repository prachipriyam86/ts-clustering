"""
embedder.py
-----------
Multimodal feature fusion for policy-relevant embeddings.

Combines two feature streams:
  - Video/audio stream: visual signal, audio signal, classifier scores
  - Context stream:     text signal, channel history, account type

The fused vector is designed to place policy-similar items
close together in embedding space — not just visually similar items.

In production, video_emb and text_emb would come from pretrained
encoders (e.g. video transformer, sentence-BERT). Here we use the
available numeric features directly to demonstrate the fusion logic.
"""

import numpy as np
from typing import List
from vertical_segmenter import ContentItem


ACCOUNT_TYPE_MAP = {
    "verified": 0.0,
    "established": 0.33,
    "new": 0.66,
    "anonymous": 1.0,
}

FLAG_TYPE_MAP = {
    "violence": 0,
    "spam": 1,
    "copyright": 2,
    "misleading": 3,
    "safe": 4,
}


class MultimodalEmbedder:
    """
    Fuses video/audio and text/context features into a
    policy-relevant embedding vector for each content item.

    Embedding dimensions:
      [0]   visual_signal          — visual modality classifier score
      [1]   audio_signal           — audio modality classifier score
      [2]   flag_confidence        — classifier confidence
      [3]   flag_type_norm         — flag type (normalized ordinal)
      [4]   text_signal            — text/metadata signal score
      [5]   channel_violation_rate — historical channel violation rate
      [6]   account_type_risk      — account type risk score

    Total: 7-dimensional policy-relevant feature vector.

    In production, dimensions [0-2] would be replaced by a
    high-dimensional video encoder output, and [4] by a
    sentence-transformer output — then projected into a shared space.
    """

    def embed(self, item: ContentItem) -> np.ndarray:
        """Return a policy-relevant embedding vector for one item."""

        # Stream 1: video + audio + classifier signals
        video_stream = np.array([
            item.visual_signal,
            item.audio_signal,
            item.flag_confidence,
            FLAG_TYPE_MAP.get(item.flag_type, 4) / len(FLAG_TYPE_MAP),
        ])

        # Stream 2: text + context signals
        context_stream = np.array([
            item.text_signal,
            item.channel_violation_rate,
            ACCOUNT_TYPE_MAP.get(item.account_type, 0.5),
        ])

        # Fuse streams (concatenation — projection layer would follow in production)
        embedding = np.concatenate([video_stream, context_stream])
        return embedding

    def embed_batch(self, items: List[ContentItem]) -> np.ndarray:
        """
        Return an (N x D) embedding matrix for a list of items.

        Args:
            items: list of ContentItem
        Returns:
            numpy array of shape (len(items), embedding_dim)
        """
        return np.vstack([self.embed(item) for item in items])
