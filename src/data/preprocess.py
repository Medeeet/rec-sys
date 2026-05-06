"""Raw JSON → clean, filtered DataFrames."""

import json
import gzip
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def load_reviews(path: Path) -> pd.DataFrame:
    """Load reviews JSON-lines into DataFrame."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                records.append(
                    {
                        "user_id": rec.get("reviewerID", ""),
                        "item_id": rec.get("asin", ""),
                        "rating": float(rec.get("overall", 0)),
                        "timestamp": int(rec.get("unixReviewTime", 0)),
                        "review_text": rec.get("reviewText", ""),
                    }
                )
            except (json.JSONDecodeError, ValueError):
                continue
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} reviews from {path.name}")
    return df


def load_metadata(path: Path) -> pd.DataFrame:
    """Load item metadata JSON-lines into DataFrame."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                # Categories: list of lists → flatten
                cats = rec.get("category", [])
                if isinstance(cats, list):
                    cats = [c for sublist in cats for c in (sublist if isinstance(sublist, list) else [sublist])]
                records.append(
                    {
                        "item_id": rec.get("asin", ""),
                        "title": rec.get("title", ""),
                        "category": " ".join(cats) if cats else "",
                        "brand": rec.get("brand", ""),
                        "description": " ".join(rec.get("description", []))
                        if isinstance(rec.get("description"), list)
                        else rec.get("description", ""),
                    }
                )
            except (json.JSONDecodeError, ValueError):
                continue
    df = pd.DataFrame(records).drop_duplicates(subset="item_id")
    logger.info(f"Loaded metadata for {len(df)} items from {path.name}")
    return df


def filter_interactions(
    df: pd.DataFrame, min_user: int = 5, min_item: int = 5, max_iter: int = 100
) -> pd.DataFrame:
    """Iterative k-core filtering: remove users/items with < N interactions."""
    prev_len = 0
    for i in range(max_iter):
        if len(df) == prev_len and i > 0:
            break
        prev_len = len(df)
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]
        # Filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]
    logger.info(
        f"After k-core filtering (k_user={min_user}, k_item={min_item}): "
        f"{len(df)} interactions, {df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items"
    )
    return df.reset_index(drop=True)


def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """Create contiguous integer ID mappings.

    Returns (user_id_map, item_id_map) where values start from 1 (0 = padding).
    """
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    user_map = {uid: i + 1 for i, uid in enumerate(unique_users)}
    item_map = {iid: i + 1 for i, iid in enumerate(unique_items)}
    logger.info(f"ID maps: {len(user_map)} users, {len(item_map)} items")
    return user_map, item_map


def preprocess_pipeline(
    reviews_path: Path,
    metadata_path: Path,
    output_dir: str = "data/processed",
    min_user: int = 5,
    min_item: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Full preprocessing pipeline.

    Returns: (interactions_df, metadata_df, user_map, item_map)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load
    reviews = load_reviews(reviews_path)
    metadata = load_metadata(metadata_path)

    # Filter
    interactions = filter_interactions(reviews, min_user, min_item)

    # Build ID maps
    user_map, item_map = build_id_maps(interactions)

    # Apply mappings
    interactions["user_idx"] = interactions["user_id"].map(user_map)
    interactions["item_idx"] = interactions["item_id"].map(item_map)

    # Filter metadata to only items in interactions
    valid_items = set(interactions["item_id"].unique())
    metadata = metadata[metadata["item_id"].isin(valid_items)].copy()
    metadata["item_idx"] = metadata["item_id"].map(item_map)

    # Sort by timestamp
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    # Save
    interactions.to_parquet(out_path / "interactions.parquet", index=False)
    metadata.to_parquet(out_path / "metadata.parquet", index=False)

    # Save ID maps
    import pickle

    with open(out_path / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(out_path / "item_map.pkl", "wb") as f:
        pickle.dump(item_map, f)

    logger.info(f"Saved processed data to {out_path}")
    return interactions, metadata, user_map, item_map
