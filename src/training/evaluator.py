"""Ranking metrics for top-K recommendation evaluation."""

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Compute Recall@K, NDCG@K, Precision@K."""

    def __init__(self, k_values: list[int] = None):
        self.k_values = k_values or [10]

    @staticmethod
    def recall_at_k(recommended: list, relevant: set, k: int) -> float:
        """Recall@K: fraction of relevant items found in top-K."""
        if not relevant:
            return 0.0
        hits = len(set(recommended[:k]) & relevant)
        return hits / len(relevant)

    @staticmethod
    def precision_at_k(recommended: list, relevant: set, k: int) -> float:
        """Precision@K: fraction of top-K that are relevant."""
        if k == 0:
            return 0.0
        hits = len(set(recommended[:k]) & relevant)
        return hits / k

    @staticmethod
    def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
        """NDCG@K: normalized discounted cumulative gain."""
        if not relevant:
            return 0.0
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # position starts from 1
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_model(
        self,
        model,
        test_loader,
        train_user_items: dict,
        n_items: int,
        device: torch.device,
        n_sample_items: int = 100,
        k: int = 10,
    ) -> dict:
        """Evaluate model with sampled metrics.

        For each test interaction, sample random negative items and
        compute ranking metrics among (pos + negatives).

        Args:
            model: trained model
            test_loader: test DataLoader
            train_user_items: {user_idx: set(item_idxs)} from training data
            n_items: total number of items
            device: torch device
            n_sample_items: number of negative items to sample per user
            k: top-K for metrics

        Returns:
            dict {Recall@K, NDCG@K, Precision@K}
        """
        model.eval()
        all_recalls = []
        all_ndcgs = []
        all_precisions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                user_ids = batch["user_idx"].to(device)
                pos_items = batch["pos_item_idx"].to(device)
                sequences = batch["sequence"].to(device)
                seq_lens = batch["seq_len"].to(device)
                pos_content = batch["pos_content"].to(device)
                context = batch["context"].to(device)

                batch_size = user_ids.size(0)

                for i in range(batch_size):
                    uid = user_ids[i].item()
                    pos_item = pos_items[i].item()
                    train_items = train_user_items.get(uid, set())

                    # Sample negative items (not in train)
                    neg_items = []
                    while len(neg_items) < n_sample_items:
                        candidate = np.random.randint(1, n_items + 1)
                        if candidate not in train_items and candidate != pos_item:
                            neg_items.append(candidate)

                    # Score positive + negatives
                    all_items = [pos_item] + neg_items
                    item_tensor = torch.tensor(all_items, dtype=torch.long, device=device)

                    # Repeat user info for all candidate items
                    u = user_ids[i].unsqueeze(0).expand(len(all_items))
                    seq = sequences[i].unsqueeze(0).expand(len(all_items), -1)
                    sl = seq_lens[i].unsqueeze(0).expand(len(all_items))
                    ctx = context[i].unsqueeze(0).expand(len(all_items), -1)

                    # For content: we'd need content for all candidate items
                    # Use pos_content repeated (simplified) — in production
                    # you'd look up each item's content
                    cnt = pos_content[i].unsqueeze(0).expand(len(all_items), -1)

                    scores = model(u, item_tensor, seq, sl, cnt, ctx)
                    # Rank items by score
                    _, indices = torch.sort(scores, descending=True)
                    ranked_items = [all_items[idx] for idx in indices.cpu().numpy()]

                    relevant = {pos_item}
                    all_recalls.append(self.recall_at_k(ranked_items, relevant, k))
                    all_ndcgs.append(self.ndcg_at_k(ranked_items, relevant, k))
                    all_precisions.append(self.precision_at_k(ranked_items, relevant, k))

        metrics = {
            f"Recall@{k}": np.mean(all_recalls),
            f"NDCG@{k}": np.mean(all_ndcgs),
            f"Precision@{k}": np.mean(all_precisions),
        }
        logger.info(f"Evaluation: {metrics}")
        return metrics
