"""Dynamic component: GRU on recent interaction sequences."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicComponent(nn.Module):
    """GRU-based sequential model capturing short-term session preferences.

    Processes the user's recent item interaction sequence through a GRU,
    extracts the final hidden state as a session representation.

    Output shape: (batch, embed_dim)
    """

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_embedding.weight[0])  # padding
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, item_sequences: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Process item sequences through GRU.

        Args:
            item_sequences: (batch, max_seq_len) padded item ID sequences
            seq_lengths: (batch,) actual sequence lengths

        Returns:
            (batch, embed_dim) session representation
        """
        # Embed items
        embedded = self.item_embedding(item_sequences)  # (batch, seq_len, embed_dim)

        # Clamp lengths to valid range
        seq_lengths = seq_lengths.clamp(min=1)

        # Pack for efficient GRU processing
        packed = pack_padded_sequence(
            embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, hidden = self.gru(packed)

        # Take final hidden state from last layer
        final_hidden = hidden[-1]  # (batch, hidden_dim)

        # Project to embed_dim
        output = self.output_proj(final_hidden)  # (batch, embed_dim)
        output = self.layer_norm(output)
        return output
