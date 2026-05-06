"""Loss functions for recommendation training."""

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    Optimizes the ranking: positive items should be scored higher than negatives.
    loss = -mean(log(sigmoid(pos_score - neg_score)))
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def bce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for point-wise training."""
    return F.binary_cross_entropy_with_logits(scores, labels)
