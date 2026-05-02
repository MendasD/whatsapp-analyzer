"""
Topic classification for WhatsApp conversation analysis.

Supports two backends selected via the `method` parameter:
  - 'lda'      — Latent Dirichlet Allocation (scikit-learn, always available).
  - 'bertopic' — BERTopic (optional; install with pip install whatsapp-analyser[bertopic]).

Input:  DataFrame produced by Cleaner  (columns include: cleaned_message)
Output: dict with keys 'df' (enriched DataFrame) and 'group_topics' (summary)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopicClassifier:
    """
    Classify messages into topics using Latent Dirichlet Allocation.

    Args:
        n_topics: Number of topics to extract.
        method:   Algorithm to use. 'lda' is always available.
                  'bertopic' requires the optional extension (issue #09).
    """

    def __init__(self, n_topics: int = 5, method: str = "lda") -> None:
        self.n_topics = n_topics
        self.method = method

    def fit_transform(self, df: pd.DataFrame) -> dict:
        """
        Fit a topic model on the cleaned DataFrame and annotate each message.

        Args:
            df: Output of Cleaner.clean(), must contain a 'cleaned_message' column.

        Returns:
            Dict with keys:
              - 'df': input DataFrame plus 'topic_id' (int) and 'topic_score' (float).
              - 'group_topics': DataFrame with columns topic_id, topic_label, weight.

        Raises:
            ValueError: If df is empty or 'cleaned_message' column is missing.
            RuntimeError: If method='bertopic' and BERTopic is not installed.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "cleaned_message" not in df.columns:
            raise ValueError("DataFrame must contain a 'cleaned_message' column.")

        texts = df["cleaned_message"].fillna("").tolist()

        if self.method == "lda":
            return self._fit_lda(df, texts)
        if self.method == "bertopic":
            return self._fit_bertopic(df, texts)

        raise ValueError(
            f"Unknown method: {self.method!r}. Supported values: 'lda', 'bertopic'."
        )

    def _fit_lda(self, df: pd.DataFrame, texts: list[str]) -> dict:
        """Vectorise with TF-IDF and fit LDA, then annotate the DataFrame."""
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info(
            "Fitting LDA with %d topics on %d messages.", self.n_topics, len(texts)
        )

        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        topic_matrix = lda.fit_transform(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()

        result_df = df.copy()
        result_df["topic_id"] = topic_matrix.argmax(axis=1).astype(int)
        result_df["topic_score"] = topic_matrix.max(axis=1).astype(float)

        group_topics = self._build_group_topics(
            lda.components_, feature_names, topic_matrix
        )

        logger.info("LDA fitting complete — %d topics extracted.", self.n_topics)
        return {"df": result_df, "group_topics": group_topics}

    def _fit_bertopic(self, df: pd.DataFrame, texts: list[str]) -> dict:
        """Fit a BERTopic model and annotate the DataFrame."""
        try:
            from bertopic import BERTopic
        except ImportError:
            raise RuntimeError(
                "BERTopic is not installed. "
                "Run: pip install 'whatsapp-analyser[bertopic]'"
            )

        logger.info(
            "Fitting BERTopic with ~%d topics on %d messages.", self.n_topics, len(texts)
        )

        topic_model = BERTopic(nr_topics=self.n_topics, calculate_probabilities=True)
        raw_topics, probs = topic_model.fit_transform(texts)

        # BERTopic assigns -1 to outliers; remap to 0 to keep topic_id non-negative
        topic_ids = [max(0, int(t)) for t in raw_topics]

        result_df = df.copy()
        result_df["topic_id"] = topic_ids
        result_df["topic_score"] = self._bertopic_scores(probs, topic_ids)

        group_topics = self._build_bertopic_group_topics(topic_model, topic_ids)

        logger.info("BERTopic fitting complete.")
        return {"df": result_df, "group_topics": group_topics}

    @staticmethod
    def _bertopic_scores(probs, topic_ids: list[int]) -> list[float]:
        """
        Extract per-document confidence scores from BERTopic probability output.

        Args:
            probs:     Probability array returned by BERTopic.fit_transform().
                       Can be 1-D (raw HDBSCAN confidences) or 2-D (n_docs × n_topics).
            topic_ids: Normalised topic assignment per document.

        Returns:
            List of floats, one score per document.
        """
        if probs is None:
            return [1.0] * len(topic_ids)
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            return [float(p) for p in probs_arr]
        # 2-D: columns correspond to topics in order; use the assigned topic's column
        scores = []
        for i, tid in enumerate(topic_ids):
            col = min(tid, probs_arr.shape[1] - 1)
            scores.append(float(probs_arr[i, col]))
        return scores

    @staticmethod
    def _build_bertopic_group_topics(topic_model, topic_ids: list[int]) -> pd.DataFrame:
        """
        Build a group_topics summary DataFrame from a fitted BERTopic model.

        Args:
            topic_model: Fitted BERTopic instance.
            topic_ids:   Normalised topic assignment per document.

        Returns:
            DataFrame with columns: topic_id, topic_label, weight.
        """
        unique_topics = sorted(set(topic_ids))
        total = len(topic_ids)
        rows = []
        for topic_id in unique_topics:
            words_scores = topic_model.get_topic(topic_id) or []
            top_words = [w for w, _ in words_scores[:5]]
            # Pad with placeholders when the model returns fewer than 5 words
            top_words += [f"word{i}" for i in range(5 - len(top_words))]
            label = " / ".join(top_words)
            weight = float(sum(1 for t in topic_ids if t == topic_id) / total)
            rows.append({"topic_id": topic_id, "topic_label": label, "weight": weight})
        return pd.DataFrame(rows)

    @staticmethod
    def _build_group_topics(
        components: np.ndarray,
        feature_names: np.ndarray,
        topic_matrix: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a topic summary DataFrame from LDA components.

        Args:
            components:    lda.components_, shape (n_topics, n_features).
            feature_names: Vocabulary array from TfidfVectorizer.
            topic_matrix:  Per-message topic distributions, shape (n_messages, n_topics).

        Returns:
            DataFrame with columns: topic_id, topic_label, weight.
        """
        assignments = topic_matrix.argmax(axis=1)
        rows = []
        for topic_id, topic_weights in enumerate(components):
            top_indices = topic_weights.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            label = " / ".join(top_words)
            weight = float((assignments == topic_id).mean())
            rows.append({"topic_id": topic_id, "topic_label": label, "weight": weight})

        return pd.DataFrame(rows)
