import torch


def make_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Args:
        token_ids: [batch_size, seq_len]
    Returns:
        mask: [batch_size, 1, 1, seq_len], True means keep.
    """
    return token_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """
    Returns:
        mask: [1, 1, seq_len, seq_len], lower-triangular causal mask.
    """
    return torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)


def make_decoder_self_attention_mask(
    tgt_input_ids: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """
    Combine target padding mask and causal mask.

    Args:
        tgt_input_ids: [batch_size, tgt_len]
    Returns:
        mask: [batch_size, 1, tgt_len, tgt_len]
    """
    batch_size, tgt_len = tgt_input_ids.size()
    padding_mask = tgt_input_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)
    causal_mask = make_causal_mask(tgt_len, tgt_input_ids.device)
    return padding_mask & causal_mask.expand(batch_size, -1, tgt_len, -1)


def make_cross_attention_mask(
    tgt_input_ids: torch.Tensor, src_ids: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """
    Encoder-decoder attention only needs to mask source padding positions.

    Args:
        tgt_input_ids: [batch_size, tgt_len]
        src_ids: [batch_size, src_len]
    Returns:
        mask: [batch_size, 1, tgt_len, src_len]
    """
    batch_size, tgt_len = tgt_input_ids.size()
    src_padding_mask = src_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)
    return src_padding_mask.expand(batch_size, 1, tgt_len, -1)

