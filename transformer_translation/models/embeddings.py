import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding without additive positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(token_ids) * self.scale)

