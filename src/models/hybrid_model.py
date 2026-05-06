"""AdaptiveHybridModel: the main dissertation contribution.

Three-component hybrid recommendation model with context-aware
attention fusion. The system adapts its recommendation strategy
based on user context (time, session state, etc.).
"""

import torch
import torch.nn as nn

from src.models.static_component import StaticComponent
from src.models.dynamic_component import DynamicComponent
from src.models.content_component import ContentComponent
from src.models.attention_gate import AttentionGate


class AdaptiveHybridModel(nn.Module):
    """Adaptive hybrid recommender with attention-based fusion.

    Components:
        1. StaticComponent (NCF) — long-term user preferences
        2. DynamicComponent (GRU) — short-term session dynamics
        3. ContentComponent (MLP) — item content features

    Fusion:
        AttentionGate produces per-sample weights based on context.
        h = α_s·h_static + α_d·h_dynamic + α_c·h_content

    Prediction:
        Score = dot(h_fused, item_embed) via prediction head.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        n_users = config["n_users"]
        n_items = config["n_items"]
        embed_dim = config.get("model", {}).get("embed_dim", 64)
        content_dim = config.get("content_dim", 5000)
        context_dim = config.get("context_dim", 5)

        # Ablation flags
        self.use_dynamic = config.get("model", {}).get("use_dynamic", True)
        self.use_content = config.get("model", {}).get("use_content", True)
        self.use_attention = config.get("model", {}).get("use_attention", True)

        # Component 1: Static (NCF)
        static_cfg = config.get("model", {}).get("static", {})
        self.static = StaticComponent(
            n_users=n_users,
            n_items=n_items,
            embed_dim=embed_dim,
            mlp_layers=static_cfg.get("mlp_layers", [128, 64]),
        )

        # Component 2: Dynamic (GRU)
        dynamic_cfg = config.get("model", {}).get("dynamic", {})
        self.dynamic = DynamicComponent(
            n_items=n_items,
            embed_dim=embed_dim,
            hidden_dim=dynamic_cfg.get("hidden_dim", 128),
            num_layers=dynamic_cfg.get("num_layers", 2),
            dropout=dynamic_cfg.get("dropout", 0.1),
        )

        # Component 3: Content (MLP)
        content_cfg = config.get("model", {}).get("content", {})
        self.content = ContentComponent(
            content_dim=content_dim,
            embed_dim=embed_dim,
            hidden_dims=content_cfg.get("hidden_dims", [512, 256]),
            dropout=content_cfg.get("dropout", 0.2),
        )

        # Fusion: Attention Gate
        attention_cfg = config.get("model", {}).get("attention", {})
        self.attention_gate = AttentionGate(
            context_dim=context_dim,
            hidden_dim=attention_cfg.get("hidden_dim", 64),
        )

        # Prediction head: fused representation -> score
        self.prediction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

        self.embed_dim = embed_dim

    def _fuse_components(
        self,
        h_static: torch.Tensor,
        h_dynamic: torch.Tensor,
        h_content: torch.Tensor,
        context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse component outputs with attention gate.

        Returns: (fused_repr, attention_weights)
        """
        # Stack: (batch, 3, embed_dim)
        components = torch.stack([h_static, h_dynamic, h_content], dim=1)

        if self.use_attention:
            weights = self.attention_gate(context_features)  # (batch, 3)
        else:
            # Uniform weights for ablation
            weights = torch.ones(
                components.size(0), 3, device=components.device
            ) / 3.0

        # Weighted sum: (batch, embed_dim)
        weights_expanded = weights.unsqueeze(-1)  # (batch, 3, 1)
        fused = (components * weights_expanded).sum(dim=1)

        return fused, weights

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_sequences: torch.Tensor,
        seq_lengths: torch.Tensor,
        item_content: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with ablation support.

        Args:
            user_ids: (batch,) user indices
            item_ids: (batch,) item indices
            item_sequences: (batch, max_seq_len) padded sequences
            seq_lengths: (batch,) actual sequence lengths
            item_content: (batch, content_dim) TF-IDF features
            context_features: (batch, context_dim) temporal features

        Returns:
            (batch,) predicted scores
        """
        # Static component: NCF on (user, item)
        h_static = self.static(user_ids, item_ids)

        # Dynamic component: GRU on sequence
        if self.use_dynamic:
            h_dynamic = self.dynamic(item_sequences, seq_lengths)
        else:
            h_dynamic = torch.zeros_like(h_static)

        # Content component: MLP on item features
        if self.use_content:
            h_content = self.content(item_content)
        else:
            h_content = torch.zeros_like(h_static)

        # Fusion
        fused, _ = self._fuse_components(
            h_static, h_dynamic, h_content, context_features
        )

        # Prediction
        score = self.prediction(fused).squeeze(-1)
        return score

    def predict_with_attention(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_sequences: torch.Tensor,
        seq_lengths: torch.Tensor,
        item_content: torch.Tensor,
        context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights for analysis.

        Returns: (scores, attention_weights)
        """
        h_static = self.static(user_ids, item_ids)

        if self.use_dynamic:
            h_dynamic = self.dynamic(item_sequences, seq_lengths)
        else:
            h_dynamic = torch.zeros_like(h_static)

        if self.use_content:
            h_content = self.content(item_content)
        else:
            h_content = torch.zeros_like(h_static)

        fused, weights = self._fuse_components(
            h_static, h_dynamic, h_content, context_features
        )

        score = self.prediction(fused).squeeze(-1)
        return score, weights

    def get_attention_weights(self, context_features: torch.Tensor) -> torch.Tensor:
        """Return attention weights for analysis (without full forward)."""
        with torch.no_grad():
            return self.attention_gate(context_features)
