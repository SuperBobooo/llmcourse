import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import build_feed_forward
from models.rope import build_position_ids


class EncoderBlock(nn.Module):
    """Pre-Norm encoder block with configurable Dense FFN or MoE FFN."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(
            d_model=config.d_model,
            num_q_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            attention_type=config.attention_type,
            rope_base=config.rope_base,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = build_feed_forward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        ffn_output, aux_loss = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)
        return x, aux_loss


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, src_len, _ = x.size()
        position_ids = build_position_ids(batch_size, src_len, x.device)
        total_aux_loss = x.new_zeros(())

        for layer in self.layers:
            x, layer_aux_loss = layer(x, src_mask=src_mask, position_ids=position_ids)
            total_aux_loss = total_aux_loss + layer_aux_loss
        return self.final_norm(x), total_aux_loss
