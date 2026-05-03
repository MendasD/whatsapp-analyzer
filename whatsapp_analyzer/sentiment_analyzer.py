"""
Sentiment analysis module for WhatsApp conversation analysis.

Supports two backends selected automatically based on language and availability:
  - VADER  — always available, language-agnostic fallback.
  - CamemBERT — optional HuggingFace model for French text (issue #06).

Input:  DataFrame produced by Cleaner  (columns include: cleaned_message, author)
Output: dict with keys 'df', 'by_user', 'global'
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Score thresholds for label assignment
_POSITIVE_THRESHOLD = 0.05
_NEGATIVE_THRESHOLD = -0.05

# CamemBERT model identifier on the HuggingFace hub
_CAMEMBERT_MODEL = "tblard/tf-allocine"


class SentimentAnalyzer:
    """
    Score the sentiment of each cleaned message.

    Args:
        lang: Language code ('fr', 'en', ...).
              When lang == 'fr' and transformers is installed, CamemBERT
              is used instead of VADER for higher accuracy on French text.
    """

    def __init__(self, lang: str = "fr") -> None:
        self.lang = lang
        # Lazily-loaded VADER analyser — initialised on first use
        self._vader: Optional[object] = None
        # Lazily-loaded CamemBERT pipeline — initialised on first use
        self._camembert: Optional[object] = None

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Score the sentiment of every message in the DataFrame.

        Args:
            df: Output of Cleaner.clean(), must contain 'cleaned_message'
                and 'author' columns.

        Returns:
            Dict with keys:
              - 'df'       — enriched DataFrame with 'sentiment_score' (float,
                             range -1 to 1) and 'sentiment_label' (str).
              - 'by_user'  — DataFrame: mean sentiment score per author.
              - 'global'   — dict: mean (float), pos_pct (float), neg_pct (float).

        Raises:
            ValueError: If df is empty or required columns are missing.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        for col in ("cleaned_message", "author"):
            if col not in df.columns:
                raise ValueError(
                    f"DataFrame must contain a '{col}' column."
                )

        texts = df["cleaned_message"].fillna("").tolist()

        if self.lang == "fr" and self._camembert_available():
            logger.info("Using CamemBERT for French sentiment analysis.")
            scores = self._score_camembert(texts)
        else:
            logger.info("Using VADER for sentiment analysis (lang=%s).", self.lang)
            scores = self._score_vader(texts)

        labels = [self._label(s) for s in scores]

        result_df = df.copy()
        result_df["sentiment_score"] = scores
        result_df["sentiment_label"] = labels

        by_user = (
            result_df.groupby("author")["sentiment_score"]
            .mean()
            .reset_index()
            .rename(columns={"sentiment_score": "mean_score"})
        )

        total = len(labels)
        pos_count = labels.count("positive")
        neg_count = labels.count("negative")
        global_stats = {
            "mean": float(result_df["sentiment_score"].mean()),
            "pos_pct": float(pos_count / total) if total else 0.0,
            "neg_pct": float(neg_count / total) if total else 0.0,
        }

        logger.info(
            "Sentiment analysis complete — mean=%.3f, pos=%.1f%%, neg=%.1f%%.",
            global_stats["mean"],
            global_stats["pos_pct"] * 100,
            global_stats["neg_pct"] * 100,
        )

        return {"df": result_df, "by_user": by_user, "global": global_stats}

    def _score_vader(self, texts: list[str]) -> list[float]:
        """
        Score each text with VADER's compound score (-1 to 1).

        The VADER analyser is instantiated once and reused across calls.
        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        if self._vader is None:
            self._vader = SentimentIntensityAnalyzer()

        return [
            float(self._vader.polarity_scores(text)["compound"])
            for text in texts
        ]

    def _score_camembert(self, texts: list[str]) -> list[float]:
        """
        Score each text using the CamemBERT-based French sentiment pipeline.

        Falls back to VADER with a log warning if transformers is unavailable
        or if loading the model fails for any reason.

        The pipeline maps HuggingFace labels to the [-1, 1] range:
          - LABEL_1 (positive) → raw probability kept as-is  (+score)
          - LABEL_0 (negative) → negated raw probability     (-score)

        Args:
            texts: List of cleaned message strings.

        Returns:
            List of float scores in [-1, 1].
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            logger.warning(
                "transformers is not installed — falling back to VADER. "
                "Run: pip install transformers torch"
            )
            return self._score_vader(texts)

        if self._camembert is None:
            try:
                self._camembert = hf_pipeline(
                    "text-classification",
                    model=_CAMEMBERT_MODEL,
                    tokenizer=_CAMEMBERT_MODEL,
                    truncation=True,
                    max_length=512,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load CamemBERT model (%s) — falling back to VADER.",
                    exc,
                )
                return self._score_vader(texts)

        scores: list[float] = []
        for text in texts:
            try:
                result = self._camembert(text)[0]
                label: str = result["label"]
                score: float = float(result["score"])
                # LABEL_1 = positive, LABEL_0 = negative
                scores.append(score if label == "LABEL_1" else -score)
            except Exception as exc:
                logger.debug("CamemBERT failed on one message (%s) — using 0.0.", exc)
                scores.append(0.0)

        return scores

    @staticmethod
    def _camembert_available() -> bool:
        """Return True when the transformers package can be imported."""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _label(score: float) -> str:
        """Map a compound score to a human-readable sentiment label."""
        if score > _POSITIVE_THRESHOLD:
            return "positive"
        if score < _NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"