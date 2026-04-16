import random
from typing import Sequence

import torch

from dataset import TranslationDataset, build_collate_fn


def _normalize_text(text: str, lowercase: bool) -> str:
    text = text.strip()
    return text.lower() if lowercase else text


def _token_set(text: str, lowercase: bool) -> set[str]:
    normalized = _normalize_text(text, lowercase=lowercase)
    return {token for token in normalized.split() if token}


def build_batch_from_pairs(
    pairs: Sequence[tuple[str, str]],
    src_tokenizer,
    tgt_tokenizer,
    src_vocab,
    tgt_vocab,
    config,
    device: str,
) -> dict:
    dataset = TranslationDataset(
        pairs=pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
    )
    collate_fn = build_collate_fn(src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    batch = collate_fn([dataset[index] for index in range(len(dataset))])
    return {
        "src_ids": batch["src_ids"].to(device),
        "tgt_input_ids": batch["tgt_input_ids"].to(device),
        "tgt_output_ids": batch["tgt_output_ids"].to(device),
        "src_texts": batch["src_texts"],
        "tgt_texts": batch["tgt_texts"],
    }


def select_retain_pairs(
    train_pairs: Sequence[tuple[str, str]],
    val_pairs: Sequence[tuple[str, str]],
    edit_source_text: str,
    retain_set_size: int,
    lowercase: bool,
    seed: int,
) -> list[tuple[str, str]]:
    edit_source_norm = _normalize_text(edit_source_text, lowercase=lowercase)
    edit_source_tokens = _token_set(edit_source_text, lowercase=lowercase)
    candidates: list[tuple[int, tuple[str, str]]] = []
    for src_text, tgt_text in list(val_pairs) + list(train_pairs):
        src_text_norm = _normalize_text(src_text, lowercase=lowercase)
        if src_text_norm == edit_source_norm:
            continue
        overlap_score = len(edit_source_tokens & _token_set(src_text, lowercase=lowercase))
        candidates.append((overlap_score, (src_text, tgt_text)))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in candidates[:retain_set_size]]


def find_matching_target(
    pairs: Sequence[tuple[str, str]],
    source_text: str,
    lowercase: bool,
) -> str | None:
    source_text_norm = _normalize_text(source_text, lowercase=lowercase)
    for src_text, tgt_text in pairs:
        if _normalize_text(src_text, lowercase=lowercase) == source_text_norm:
            return tgt_text
    return None


def ensure_text_known(text: str, tokenizer, vocab, side: str) -> list[str]:
    tokens = tokenizer.tokenize(text)
    if not tokens:
        raise ValueError(f"{side} text is empty after tokenization: {text!r}")

    unknown_tokens = [token for token in tokens if vocab.token_to_id(token) == vocab.unk_id]
    if unknown_tokens:
        raise ValueError(
            f"{side} text contains tokens outside the current vocabulary: {unknown_tokens}. "
            "For this simplified editing experiment, please choose a text whose tokens "
            "already exist in the trained vocabulary."
        )
    return tokens
