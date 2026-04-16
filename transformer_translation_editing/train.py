import json
import os
from dataclasses import asdict
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs import Config, get_config
from dataset import TranslationDataset, build_collate_fn, read_parallel_corpus
from models import TransformerTranslationModel
from utils import (
    BasicTokenizer,
    Vocab,
    make_cross_attention_mask,
    make_decoder_self_attention_mask,
    make_padding_mask,
    save_loss_curve,
    save_translation_samples,
    set_seed,
    token_accuracy,
)


def build_vocabs(
    train_pairs: list[tuple[str, str]],
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    config: Config,
) -> tuple[Vocab, Vocab]:
    src_tokenized = (src_tokenizer.tokenize(src_text) for src_text, _ in train_pairs)
    tgt_tokenized = (tgt_tokenizer.tokenize(tgt_text) for _, tgt_text in train_pairs)

    src_vocab = Vocab.build(
        src_tokenized,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    tgt_vocab = Vocab.build(
        tgt_tokenized,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    src_vocab.save(config.src_vocab_path)
    tgt_vocab.save(config.tgt_vocab_path)
    return src_vocab, tgt_vocab


def load_dataset_metadata(config: Config) -> dict | None:
    if not config.dataset_metadata_path.exists():
        return None
    with config.dataset_metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_dataset_status(config: Config, metadata: dict | None) -> None:
    if metadata is None:
        print(
            "Dataset metadata not found. Training will use the existing TSV files under "
            "data/raw/. Run scripts/prepare_dataset.py first if you want the formal "
            "IWSLT2017 or Multi30k setup."
        )
        return

    prepared_name = metadata.get("dataset_name", "unknown")
    if prepared_name != config.dataset_name:
        print(
            f"Warning: config.dataset_name={config.dataset_name}, but the prepared TSV "
            f"files were generated from {prepared_name}. Training will continue with "
            "the current TSV files."
        )

    print(
        "Prepared dataset metadata: "
        f"dataset={prepared_name}, "
        f"train={metadata.get('train_samples', 'n/a')}, "
        f"val={metadata.get('val_samples', 'n/a')}, "
        f"test={metadata.get('test_samples', 'n/a')}, "
        f"max_src_len={metadata.get('max_src_len', 'n/a')}, "
        f"max_tgt_len={metadata.get('max_tgt_len', 'n/a')}"
    )


def build_dataloaders(
    config: Config,
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
) -> tuple[DataLoader, DataLoader, TranslationDataset, TranslationDataset]:
    train_pairs = read_parallel_corpus(config.train_path)
    val_pairs = read_parallel_corpus(config.val_path)

    train_dataset = TranslationDataset(
        pairs=train_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
    )
    val_dataset = TranslationDataset(
        pairs=val_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
    )

    collate_fn = build_collate_fn(
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_dataset, val_dataset


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


def move_batch_to_device(batch: dict, device: str) -> dict:
    return {
        "src_ids": batch["src_ids"].to(device),
        "tgt_input_ids": batch["tgt_input_ids"].to(device),
        "tgt_output_ids": batch["tgt_output_ids"].to(device),
        "src_texts": batch["src_texts"],
        "tgt_texts": batch["tgt_texts"],
    }


def run_epoch(
    model: TransformerTranslationModel,
    data_loader: DataLoader,
    optimizer: AdamW | None,
    src_pad_id: int,
    tgt_pad_id: int,
    label_smoothing: float,
    clip_grad_norm: float,
    device: str,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_acc = 0.0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device=device)
        src_ids = batch["src_ids"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_output_ids = batch["tgt_output_ids"]

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

        if is_training:
            optimizer.zero_grad()

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
            label_smoothing=label_smoothing,
        )

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        total_acc += token_accuracy(logits.detach(), tgt_output_ids, pad_id=tgt_pad_id)

    num_batches = max(1, len(data_loader))
    return total_loss / num_batches, total_acc / num_batches


def ids_to_text(token_ids: list[int], vocab: Vocab, tokenizer: BasicTokenizer) -> str:
    tokens = []
    for token_id in token_ids:
        token = vocab.id_to_token(token_id)
        if token == Vocab.BOS_TOKEN:
            continue
        if token in {Vocab.EOS_TOKEN, Vocab.PAD_TOKEN}:
            break
        tokens.append(token)
    return tokenizer.detokenize(tokens)


@torch.no_grad()
def generate_translation_samples(
    model: TransformerTranslationModel,
    dataset: TranslationDataset,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    tgt_tokenizer: BasicTokenizer,
    config: Config,
    num_samples: int,
) -> list[dict]:
    model.eval()
    samples = []
    sample_count = min(num_samples, len(dataset))

    for index in range(sample_count):
        item = dataset[index]
        src_ids = item["src_ids"].unsqueeze(0).to(config.device)
        src_mask = make_padding_mask(src_ids, pad_id=src_vocab.pad_id)

        generated_ids = model.greedy_decode(
            src_ids=src_ids,
            src_mask=src_mask,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=config.max_decode_len,
        )[0].tolist()

        samples.append(
            {
                "source": item["src_text"],
                "reference": item["tgt_text"],
                "generated": ids_to_text(generated_ids, tgt_vocab, tgt_tokenizer),
            }
        )
    return samples


def train_model(config: Config | None = None) -> dict:
    config = config or get_config()
    set_seed(config.seed)
    config.ensure_dirs()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    src_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    tgt_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    dataset_metadata = load_dataset_metadata(config)
    print_dataset_status(config, dataset_metadata)

    train_pairs = read_parallel_corpus(config.train_path)
    if not train_pairs:
        raise RuntimeError(
            "Training corpus is empty. Please prepare data/raw/train.tsv first. "
            "Recommended command: python scripts/prepare_dataset.py "
            f"--dataset_name {config.dataset_name}"
        )

    src_vocab, tgt_vocab = build_vocabs(
        train_pairs=train_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
    )
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        config=config,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    model = TransformerTranslationModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=config,
    ).to(config.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {config.device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Source vocab size: {len(src_vocab)} | Target vocab size: {len(tgt_vocab)}")
    print(
        f"Attention type: {config.attention_type} | "
        f"num_q_heads={config.num_q_heads} | num_kv_heads={config.num_kv_heads}"
    )
    print(
        f"Batch size: {config.batch_size} | Epochs: {config.epochs} | "
        f"Early stopping patience: {config.early_stopping_patience}"
    )

    for epoch in range(1, config.epochs + 1):
        start_time = perf_counter()

        # Teacher forcing:
        # decoder input is tgt_input_ids = [<bos>, y1, y2, ...]
        # supervision target is tgt_output_ids = [y1, y2, ..., <eos>]
        train_loss, train_acc = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
            label_smoothing=config.label_smoothing,
            clip_grad_norm=config.clip_grad_norm,
            device=config.device,
        )

        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                src_pad_id=src_vocab.pad_id,
                tgt_pad_id=tgt_vocab.pad_id,
                label_smoothing=config.label_smoothing,
                clip_grad_norm=config.clip_grad_norm,
                device=config.device,
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = perf_counter() - start_time
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}."
            )
            break

    save_loss_curve(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        output_path=config.loss_curve_path,
    )

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)

    sample_dataset = val_dataset if len(val_dataset) > 0 else train_dataset
    samples = generate_translation_samples(
        model=model,
        dataset=sample_dataset,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tgt_tokenizer=tgt_tokenizer,
        config=config,
        num_samples=config.sample_count,
    )
    save_translation_samples(samples, config.sample_output_path)

    return {
        "model": model,
        "config": config,
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "history": history,
        "samples": samples,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


if __name__ == "__main__":
    train_model()
