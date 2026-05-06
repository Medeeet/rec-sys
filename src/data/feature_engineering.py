"""Feature engineering: TF-IDF, user sequences, temporal features."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Builds and caches all feature matrices."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build_item_content_features(
        self, metadata_df: pd.DataFrame, max_features: int = 5000
    ) -> np.ndarray:
        """Build TF-IDF features from item title + category text.

        Returns dense matrix of shape (max_item_idx + 1, max_features).
        Index 0 is reserved for padding (zero vector).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        cache_path = self.processed_dir / "item_content_features.npy"
        if cache_path.exists():
            logger.info(f"Loading cached item content features from {cache_path}")
            return np.load(cache_path)

        # Combine title + category as text
        metadata_df = metadata_df.copy()
        metadata_df["text"] = (
            metadata_df["title"].fillna("")
            + " "
            + metadata_df["category"].fillna("")
        )

        max_idx = int(metadata_df["item_idx"].max())
        logger.info(f"Building TF-IDF features for {len(metadata_df)} items, max_features={max_features}")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(metadata_df["text"])

        # Create aligned dense matrix (item_idx -> features)
        features = np.zeros((max_idx + 1, tfidf_matrix.shape[1]), dtype=np.float32)
        for i, idx in enumerate(metadata_df["item_idx"].values):
            features[int(idx)] = tfidf_matrix[i].toarray().squeeze()

        np.save(cache_path, features)
        logger.info(f"Item content features shape: {features.shape}, saved to {cache_path}")
        return features

    def build_user_sequences(
        self,
        interactions_df: pd.DataFrame,
        max_length: int = 50,
    ) -> dict[int, np.ndarray]:
        """Build per-user item sequences ordered by time.

        Returns: {user_idx: np.array([item_idx_1, item_idx_2, ...])}
        Sequences are truncated to the last max_length items.
        """
        cache_path = self.processed_dir / "user_sequences.npz"
        if cache_path.exists():
            logger.info(f"Loading cached user sequences from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return dict(data["sequences"].item())

        logger.info("Building user sequences...")
        sorted_df = interactions_df.sort_values(["user_idx", "timestamp"])
        sequences = {}
        for user_idx, group in sorted_df.groupby("user_idx"):
            items = group["item_idx"].values
            # Keep last max_length items
            if len(items) > max_length:
                items = items[-max_length:]
            sequences[int(user_idx)] = items.astype(np.int64)

        np.savez(cache_path, sequences=sequences)
        logger.info(f"Built sequences for {len(sequences)} users")
        return sequences

    def build_temporal_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical temporal features from timestamps.

        Returns DataFrame with columns:
            hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
        """
        cache_path = self.processed_dir / "temporal_features.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached temporal features from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info("Building temporal features...")
        dt = pd.to_datetime(interactions_df["timestamp"], unit="s")

        features = pd.DataFrame(index=interactions_df.index)

        # Cyclical encoding for hour of day
        hour = dt.dt.hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

        # Cyclical encoding for day of week
        dow = dt.dt.dayofweek
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

        # Binary weekend flag
        features["is_weekend"] = (dow >= 5).astype(np.float32)

        features.to_parquet(cache_path, index=False)
        logger.info(f"Temporal features shape: {features.shape}")
        return features
