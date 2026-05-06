"""Temporal train/val/test splitting."""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically (global temporal split).

    All train timestamps < all val timestamps < all test timestamps.
    This prevents temporal data leakage.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    logger.info(
        f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    logger.info(
        f"Time ranges: train=[{train_df[timestamp_col].min()}-{train_df[timestamp_col].max()}], "
        f"val=[{val_df[timestamp_col].min()}-{val_df[timestamp_col].max()}], "
        f"test=[{test_df[timestamp_col].min()}-{test_df[timestamp_col].max()}]"
    )
    return train_df, val_df, test_df
