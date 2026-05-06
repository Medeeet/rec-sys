"""Content-based component: MLP on item feature vectors (TF-IDF)."""

import torch
import torch.nn as nn


class ContentComponent(nn.Module):
    """MLP processing pre-computed item content features.

    Takes TF-IDF vectors and produces dense item representations.
    Crucial for cold-start items that have no interaction history
    but do have text descriptions/categories.

    Output shape: (batch, embed_dim)
    """

    def __init__(
        self,
        content_dim: int,
        embed_dim: int = 64,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = content_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, item_content: torch.Tensor) -> torch.Tensor:
        """Process item content features.

        Args:
            item_content: (batch, content_dim) TF-IDF feature vectors

        Returns:
            (batch, embed_dim) content-aware item representation
        """
        return self.mlp(item_content)
