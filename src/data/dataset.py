"""PyTorch Dataset and DataLoader for the hybrid model."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import get_logger

logger = get_logger(__name__)


def _get_content_row(item_content, idx: int, content_dim: int) -> np.ndarray:
    """Get item content vector — handles both dense (ndarray) and sparse (scipy) matrices."""
    if idx >= item_content.shape[0]:
        return np.zeros(content_dim, dtype=np.float32)
    row = item_content[idx]
    # scipy sparse row → dense
    if hasattr(row, "toarray"):
        return row.toarray().squeeze().astype(np.float32)
    return np.asarray(row, dtype=np.float32)


class RecommendationDataset(Dataset):
    """BPR-style triplet dataset for the hybrid model.

    Each sample provides:
    - user_idx, pos_item_idx, neg_item_idx
    - user_sequence (padded item history for GRU)
    - item_content (TF-IDF vector for positive item)
    - context_features (temporal features)

    item_content_features can be either:
    - np.ndarray  (dense, shape: n_items+1 × content_dim)
    - scipy sparse matrix (same shape, much less RAM)
    """

    def __init__(
        self,
        interactions_df,
        user_sequences: dict,
        item_content_features,
        temporal_features_df,
        n_items: int,
        max_seq_len: int = 50,
        negative_sampling: bool = True,
    ):
        self.interactions = interactions_df.reset_index(drop=True)
        self.user_sequences = user_sequences
        self.item_content = item_content_features
        self.content_dim = item_content_features.shape[1]
        self.temporal_features = temporal_features_df.values.astype(np.float32)
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.negative_sampling = negative_sampling

        # Build user->items set for negative sampling
        self.user_items = {}
        uid_arr = self.interactions["user_idx"].values
        iid_arr = self.interactions["item_idx"].values
        for uid, iid in zip(uid_arr, iid_arr):
            self.user_items.setdefault(int(uid), set()).add(int(iid))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_idx = int(row["user_idx"])
        pos_item_idx = int(row["item_idx"])

        # Negative sampling
        if self.negative_sampling:
            user_set = self.user_items.get(user_idx, set())
            neg_item_idx = np.random.randint(1, self.n_items + 1)
            while neg_item_idx in user_set:
                neg_item_idx = np.random.randint(1, self.n_items + 1)
        else:
            neg_item_idx = 0

        # User sequence (padded, right-aligned)
        seq = self.user_sequences.get(user_idx, np.array([], dtype=np.int64))
        seq_len = min(len(seq), self.max_seq_len)
        padded_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        if seq_len > 0:
            padded_seq[-seq_len:] = seq[-seq_len:]

        # Item content (dense or sparse → always returns np.float32 vector)
        pos_content = _get_content_row(self.item_content, pos_item_idx, self.content_dim)
        neg_content = _get_content_row(self.item_content, neg_item_idx, self.content_dim)

        # Context
        context = self.temporal_features[idx] if idx < len(self.temporal_features) else np.zeros(5, dtype=np.float32)

        return {
            "user_idx":     torch.tensor(user_idx,     dtype=torch.long),
            "pos_item_idx": torch.tensor(pos_item_idx, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg_item_idx, dtype=torch.long),
            "sequence":     torch.tensor(padded_seq,   dtype=torch.long),
            "seq_len":      torch.tensor(max(seq_len, 1), dtype=torch.long),
            "pos_content":  torch.tensor(pos_content,  dtype=torch.float32),
            "neg_content":  torch.tensor(neg_content,  dtype=torch.float32),
            "context":      torch.tensor(context,      dtype=torch.float32),
        }


def get_dataloaders(
    train_df,
    val_df,
    test_df,
    user_sequences,
    item_content_features,
    train_temporal,
    val_temporal,
    test_temporal,
    n_items,
    config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders."""
    max_seq_len = config.get("model", {}).get("dynamic", {}).get("max_seq_len", 50)
    batch_size = config.get("training", {}).get("batch_size", 1024)

    train_ds = RecommendationDataset(
        train_df, user_sequences, item_content_features, train_temporal,
        n_items, max_seq_len, negative_sampling=True,
    )
    val_ds = RecommendationDataset(
        val_df, user_sequences, item_content_features, val_temporal,
        n_items, max_seq_len, negative_sampling=True,
    )
    test_ds = RecommendationDataset(
        test_df, user_sequences, item_content_features, test_temporal,
        n_items, max_seq_len, negative_sampling=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, batch_size={batch_size}")
    return train_loader, val_loader, test_loader
