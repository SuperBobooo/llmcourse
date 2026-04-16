import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward
from models.rope import build_position_ids


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block."""

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
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            attention_type=attention_type,
            rope_base=rope_base,
        )
        self.norm3 = nn.LayerNorm(d_model)
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
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_input = self.norm1(x)
        self_attn_output, _ = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            attn_mask=self_attn_mask,
            use_rope=True,
            position_ids=position_ids,
        )
        x = x + self.dropout(self_attn_output)

        cross_attn_input = self.norm2(x)
        cross_attn_output, _ = self.cross_attn(
            cross_attn_input,
            memory,
            memory,
            attn_mask=cross_attn_mask,
            use_rope=False,
        )
        # Cross-attention does not use RoPE because decoder queries and encoder
        # keys belong to different sequences. Their positional frames are not shared.
        x = x + self.dropout(cross_attn_output)

        ffn_input = self.norm3(x)
        x = x + self.dropout(self.ffn(ffn_input))
        return x


class Decoder(nn.Module):
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
                DecoderBlock(
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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = x.size()
        position_ids = build_position_ids(batch_size, tgt_len, x.device)

        for layer in self.layers:
            x = layer(
                x,
                memory=memory,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                position_ids=position_ids,
            )
        return self.final_norm(x)

