import math

import torch
import torch.nn as nn

from models.rope import RotaryEmbedding, build_position_ids


class ScaledDotProductAttention(nn.Module):
    """Manual scaled dot-product attention."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Unified attention module for:
        - MHA: num_kv_heads == num_q_heads
        - GQA: 1 < num_kv_heads < num_q_heads
        - MQA: num_kv_heads == 1

    In the old project, GQA was already implemented explicitly, and MQA worked
    implicitly when attention_type='gqa' with num_kv_heads=1. Here it is
    normalized into a clear three-mode interface.
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout: float,
        attention_type: str = "gqa",
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads.")

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.attention_type = attention_type.lower()
        self.head_dim = d_model // num_q_heads

        if self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even head_dim. Please adjust d_model / num_q_heads."
            )

        self.num_kv_heads = self._normalize_num_kv_heads(
            attention_type=self.attention_type,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )
        self.num_query_per_kv = self.num_q_heads // self.num_kv_heads

        self.q_proj = nn.Linear(d_model, self.num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(head_dim=self.head_dim, base=rope_base)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _normalize_num_kv_heads(
        self,
        attention_type: str,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> int:
        if attention_type == "mha":
            return num_q_heads
        if attention_type == "mqa":
            return 1
        if attention_type == "gqa":
            if num_q_heads < num_kv_heads:
                raise ValueError("GQA requires num_q_heads >= num_kv_heads.")
            if num_q_heads % num_kv_heads != 0:
                raise ValueError("num_q_heads must be divisible by num_kv_heads.")
            return num_kv_heads
        raise ValueError("attention_type must be one of: mha, gqa, mqa")

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _reshape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def _repeat_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_q_heads == self.num_kv_heads:
            return x
        return x.repeat_interleave(self.num_query_per_kv, dim=1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = False,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_len, _ = query.size()
        _, k_len, _ = key.size()

        q = self._reshape_q(self.q_proj(query))
        k = self._reshape_kv(self.k_proj(key))
        v = self._reshape_kv(self.v_proj(value))

        if use_rope:
            if q_len != k_len:
                raise ValueError("RoPE is only used in self-attention where q_len == k_len.")
            if position_ids is None:
                position_ids = build_position_ids(batch_size, q_len, query.device)

            # Required order:
            # 1. Linear projection to q / k / v
            # 2. Reshape into heads
            # 3. Apply RoPE to q / k in self-attention
            # 4. Expand KV heads for GQA / MQA
            q, k = self.rope.apply_to_qk(q, k, position_ids)

        if self.num_q_heads != self.num_kv_heads:
            k = self._repeat_kv_heads(k)
            v = self._repeat_kv_heads(v)

        attn_output, attn_weights = self.attention(q, k, v, attn_mask=attn_mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        )
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
