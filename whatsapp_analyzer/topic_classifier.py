"""
Topic classification for WhatsApp conversation analysis.

Assigns a topic to each message using Latent Dirichlet Allocation (LDA)
from scikit-learn. BERTopic support is provided by a separate extension
(issue #09).

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
        """Route to BERTopic when the optional dependency is available."""
        try:
            from bertopic import BERTopic  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "BERTopic is not installed. "
                "Run: pip install 'whatsapp-analyser[bertopic]'"
            )
        # Full implementation added in issue #09.
        raise NotImplementedError("BERTopic support is implemented in issue #09.")

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
