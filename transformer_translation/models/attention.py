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
        """
        Args:
            q: [batch_size, num_heads, q_len, head_dim]
            k: [batch_size, num_heads, k_len, head_dim]
            v: [batch_size, num_heads, k_len, head_dim]
            attn_mask: [batch_size, 1 or num_heads, q_len, k_len], True means keep.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Manual multi-head attention with support for:
        - standard MHA
        - GQA (Grouped Query Attention)

    RoPE is optionally applied only to self-attention q/k before KV expansion.
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

        if self.attention_type not in {"mha", "gqa"}:
            raise ValueError("attention_type must be 'mha' or 'gqa'.")

        if self.attention_type == "mha":
            self.num_kv_heads = num_q_heads
        else:
            if num_q_heads < num_kv_heads:
                raise ValueError("GQA requires num_q_heads >= num_kv_heads.")
            if num_q_heads % num_kv_heads != 0:
                raise ValueError("num_q_heads must be divisible by num_kv_heads.")
            self.num_kv_heads = num_kv_heads

        self.num_query_per_kv = self.num_q_heads // self.num_kv_heads

        self.q_proj = nn.Linear(d_model, self.num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(head_dim=self.head_dim, base=rope_base)
        self.attention = ScaledDotProductAttention(dropout=dropout)

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
        """
        Args:
            query: [batch_size, q_len, d_model]
            key: [batch_size, k_len, d_model]
            value: [batch_size, k_len, d_model]
        """
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

            # The required order is:
            # 1) linear projection
            # 2) reshape into heads
            # 3) apply RoPE to q/k
            # 4) expand/repeat KV heads for GQA
            q, k = self.rope.apply_to_qk(q, k, position_ids)

        if self.attention_type == "gqa":
            k = self._repeat_kv_heads(k)
            v = self._repeat_kv_heads(v)

        attn_output, attn_weights = self.attention(q, k, v, attn_mask=attn_mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        )
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

