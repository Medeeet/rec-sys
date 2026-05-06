"""Static component: Neural Collaborative Filtering (NCF) for long-term preferences."""

import torch
import torch.nn as nn


class StaticComponent(nn.Module):
    """NCF combining GMF and MLP paths.

    - GMF: element-wise product of user/item embeddings
    - MLP: concatenated embeddings through dense layers
    - Output: combined representation of shape (batch, embed_dim)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        mlp_layers: list[int] = None,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64]

        # GMF path embeddings
        self.user_embedding_gmf = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.item_embedding_gmf = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)

        # MLP path embeddings
        self.user_embedding_mlp = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.item_embedding_mlp = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)

        # MLP layers
        layers = []
        in_dim = embed_dim * 2
        for h_dim in mlp_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        # Combine GMF + MLP
        self.combine = nn.Linear(embed_dim + mlp_layers[-1], embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute NCF user-item compatibility.

        Args:
            user_ids: (batch,) user indices
            item_ids: (batch,) item indices

        Returns:
            (batch, embed_dim) compatibility representation
        """
        # GMF path: element-wise product
        u_gmf = self.user_embedding_gmf(user_ids)
        i_gmf = self.item_embedding_gmf(item_ids)
        gmf_out = u_gmf * i_gmf  # (batch, embed_dim)

        # MLP path: concatenate and forward
        u_mlp = self.user_embedding_mlp(user_ids)
        i_mlp = self.item_embedding_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)  # (batch, 2*embed_dim)
        mlp_out = self.mlp(mlp_in)  # (batch, mlp_layers[-1])

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.combine(combined)  # (batch, embed_dim)
        return output
