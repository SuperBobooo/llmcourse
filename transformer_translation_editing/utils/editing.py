import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import get_config
from dataset import TranslationDataset, build_collate_fn, read_parallel_corpus
from models import TransformerTranslationModel
from utils import (
    BasicTokenizer,
    Vocab,
    make_cross_attention_mask,
    make_decoder_self_attention_mask,
    make_padding_mask,
    token_accuracy,
)


MODEL_CONFIG_KEYS = {
    "d_model",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_q_heads",
    "num_kv_heads",
    "attention_type",
    "d_ff",
    "dropout",
    "ffn_activation",
    "rope_base",
    "lowercase",
    "max_src_len",
    "max_tgt_len",
    "max_decode_len",
}


def apply_saved_model_config(config, saved_config: dict | None):
    if not saved_config:
        return config

    for key, value in saved_config.items():
        if key in MODEL_CONFIG_KEYS and hasattr(config, key):
            setattr(config, key, value)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.ensure_dirs()
    return config


def load_checkpoint_bundle(
    checkpoint_path: Path | str,
    config=None,
) -> tuple[TransformerTranslationModel, BasicTokenizer, BasicTokenizer, Vocab, Vocab, object, dict]:
    config = config or get_config()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    config = apply_saved_model_config(config, checkpoint.get("config"))

    if not config.src_vocab_path.exists() or not config.tgt_vocab_path.exists():
        raise FileNotFoundError(
            "Vocabulary files are missing. Please train the base model first."
        )

    src_vocab = Vocab.load(config.src_vocab_path)
    tgt_vocab = Vocab.load(config.tgt_vocab_path)

    model = TransformerTranslationModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    src_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    tgt_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    return model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, config, checkpoint


def decode_generated_ids(generated_ids: list[int], tgt_vocab: Vocab, tgt_tokenizer: BasicTokenizer) -> str:
    tokens = []
    for token_id in generated_ids:
        token = tgt_vocab.id_to_token(token_id)
        if token == Vocab.BOS_TOKEN:
            continue
        if token in {Vocab.EOS_TOKEN, Vocab.PAD_TOKEN}:
            break
        tokens.append(token)
    return tgt_tokenizer.detokenize(tokens)


@torch.no_grad()
def translate_sentence(
    sentence: str,
    model: TransformerTranslationModel,
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config,
) -> str:
    src_tokens = src_tokenizer.tokenize(sentence)[: config.max_src_len - 1]
    src_ids = torch.tensor(
        src_vocab.encode(src_tokens, add_eos=True),
        dtype=torch.long,
        device=config.device,
    ).unsqueeze(0)
    src_mask = make_padding_mask(src_ids, pad_id=src_vocab.pad_id)

    generated_ids = model.greedy_decode(
        src_ids=src_ids,
        src_mask=src_mask,
        bos_id=tgt_vocab.bos_id,
        eos_id=tgt_vocab.eos_id,
        max_len=config.max_decode_len,
    )[0].tolist()
    return decode_generated_ids(generated_ids, tgt_vocab, tgt_tokenizer)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def build_eval_loader(config, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab) -> DataLoader:
    val_pairs = read_parallel_corpus(config.val_path)
    dataset = TranslationDataset(
        pairs=val_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
    )
    collate_fn = build_collate_fn(src_pad_id=src_vocab.pad_id, tgt_pad_id=tgt_vocab.pad_id)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate_model_on_loader(model, data_loader: DataLoader, src_pad_id: int, tgt_pad_id: int, config) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in data_loader:
        src_ids = batch["src_ids"].to(config.device)
        tgt_input_ids = batch["tgt_input_ids"].to(config.device)
        tgt_output_ids = batch["tgt_output_ids"].to(config.device)

        src_mask = make_padding_mask(src_ids, pad_id=src_pad_id)
        tgt_self_attn_mask = make_decoder_self_attention_mask(
            tgt_input_ids,
            pad_id=tgt_pad_id,
        )
        cross_attn_mask = make_cross_attention_mask(
            tgt_input_ids,
            src_ids,
            pad_id=src_pad_id,
        )

        logits = model(
            src_ids=src_ids,
            tgt_input_ids=tgt_input_ids,
            src_mask=src_mask,
            tgt_self_attn_mask=tgt_self_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )
        loss = compute_loss(
            logits=logits,
            targets=tgt_output_ids,
            pad_id=tgt_pad_id,
            label_smoothing=config.label_smoothing,
        )
        total_loss += loss.item()
        total_acc += token_accuracy(logits, tgt_output_ids, pad_id=tgt_pad_id)

    num_batches = max(1, len(data_loader))
    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
    }


def save_json(payload: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def translation_contains_unk(text: str) -> bool:
    return Vocab.UNK_TOKEN in text.split()


def assess_edit_case(before_translation: str) -> dict:
    normalized = before_translation.strip()
    contains_unk = translation_contains_unk(normalized)
    is_existing_knowledge_case = bool(normalized) and not contains_unk

    if is_existing_knowledge_case:
        note = "编辑前已经是完整翻译，更符合“修改已有知识”的实验设定。"
        case_type = "existing_knowledge_edit"
    else:
        note = "当前案例更偏向新增映射或部分补全，因为编辑前输出仍包含 <unk>。"
        case_type = "mapping_completion_or_partial_edit"

    return {
        "before_translation": normalized,
        "contains_unk": contains_unk,
        "is_existing_knowledge_case": is_existing_knowledge_case,
        "case_type": case_type,
        "note": note,
    }
