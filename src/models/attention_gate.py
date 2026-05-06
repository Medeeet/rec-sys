"""Context-aware attention gate for adaptive component fusion.

This is the key "adaptive" mechanism of the dissertation:
different users in different contexts get different blends
of static/dynamic/content signals.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Learns context-dependent weights for three model components.

    Input: context vector (temporal features + optional user stats)
    Output: 3 attention weights (α_static, α_dynamic, α_content)

    The final fused representation:
        h = α_s · h_static + α_d · h_dynamic + α_c · h_content
    """

    def __init__(self, context_dim: int, hidden_dim: int = 64, n_components: int = 3):
        super().__init__()
        self.n_components = n_components

        self.gate_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_components),
        )

        # Temperature parameter for softmax sharpness
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Initialize last layer bias to uniform weights
        last_linear = [m for m in self.gate_network if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute attention weights from context.

        Args:
            context: (batch, context_dim) contextual features

        Returns:
            weights: (batch, n_components) attention weights summing to 1
        """
        logits = self.gate_network(context)  # (batch, n_components)
        # Temperature-scaled softmax
        weights = torch.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        return weights
