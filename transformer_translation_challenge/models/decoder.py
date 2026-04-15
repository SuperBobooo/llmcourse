import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import build_feed_forward
from models.rope import build_position_ids


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block with configurable Dense FFN or MoE FFN."""

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
        self.cross_attn = MultiHeadAttention(
            d_model=config.d_model,
            num_q_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            attention_type=config.attention_type,
            rope_base=config.rope_base,
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        self.ffn = build_feed_forward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # Cross-attention keeps standard attention semantics and does not use RoPE.
        x = x + self.dropout(cross_attn_output)

        ffn_input = self.norm3(x)
        ffn_output, aux_loss = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)
        return x, aux_loss


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, tgt_len, _ = x.size()
        position_ids = build_position_ids(batch_size, tgt_len, x.device)
        total_aux_loss = x.new_zeros(())

        for layer in self.layers:
            x, layer_aux_loss = layer(
                x,
                memory=memory,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                position_ids=position_ids,
            )
            total_aux_loss = total_aux_loss + layer_aux_loss
        return self.final_norm(x), total_aux_loss
