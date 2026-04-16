import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward
from models.rope import build_position_ids


class EncoderBlock(nn.Module):
    """Pre-Norm encoder block."""

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float,
        attention_type: str,
        activation: str,
        rope_base: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            attention_type=attention_type,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=src_mask,
            use_rope=True,
            position_ids=position_ids,
        )
        x = x + self.dropout(attn_output)

        ffn_input = self.norm2(x)
        x = x + self.dropout(self.ffn(ffn_input))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float,
        attention_type: str,
        activation: str,
        rope_base: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_type=attention_type,
                    activation=activation,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, src_len, _ = x.size()
        position_ids = build_position_ids(batch_size, src_len, x.device)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask, position_ids=position_ids)
        return self.final_norm(x)

