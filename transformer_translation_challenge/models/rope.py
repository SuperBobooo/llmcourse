import torch
import torch.nn as nn


def build_position_ids(
    batch_size: int, seq_len: int, device: torch.device | str
) -> torch.Tensor:
    """
    Create position ids for each token position.

    Returns:
        position_ids: [batch_size, seq_len]
    """
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
        batch_size, -1
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE rotates the last dimension by pairing even/odd channels.

    Args:
        x: [batch_size, num_heads, seq_len, head_dim]
    """
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires head_dim to be even.")

    x_even = x[..., : head_dim // 2]
    x_odd = x[..., head_dim // 2 :]
    return torch.cat((-x_odd, x_even), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding directly on q/k.

    Args:
        x: [batch_size, num_heads, seq_len, head_dim]
        cos: [batch_size, 1, seq_len, head_dim]
        sin: [batch_size, 1, seq_len, head_dim]
    """
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding.

    RoPE is not added to token embeddings. Instead, it rotates q and k inside
    self-attention so that position information is encoded in the relative phase.
    """

    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, but got head_dim={head_dim}."
            )

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self.max_seq_len_cached = 0

    def _build_cache(
        self, seq_len: int, device: torch.device | str, dtype: torch.dtype
    ) -> None:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.cos_cache = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        self.sin_cache = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        self.max_seq_len_cached = seq_len

    def get_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = position_ids.size()
        device = position_ids.device

        cache_invalid = (
            self.cos_cache.numel() == 0
            or self.max_seq_len_cached < seq_len
            or self.cos_cache.device != device
            or self.cos_cache.dtype != dtype
        )
        if cache_invalid:
            self._build_cache(seq_len, device, dtype)

        flat_positions = position_ids.reshape(-1)
        base_cos = self.cos_cache[0, 0]
        base_sin = self.sin_cache[0, 0]

        cos = base_cos.index_select(0, flat_positions).view(batch_size, seq_len, -1)
        sin = base_sin.index_select(0, flat_positions).view(batch_size, seq_len, -1)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def apply_to_qk(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.get_cos_sin(position_ids, dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

