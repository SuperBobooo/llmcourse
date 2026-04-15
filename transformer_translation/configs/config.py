import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent.parent
    train_path: Path = project_root / "data" / "raw" / "train.tsv"
    val_path: Path = project_root / "data" / "raw" / "val.tsv"
    processed_dir: Path = project_root / "data" / "processed"
    src_vocab_path: Path = processed_dir / "src_vocab.json"
    tgt_vocab_path: Path = processed_dir / "tgt_vocab.json"
    outputs_dir: Path = project_root / "outputs"
    checkpoint_dir: Path = outputs_dir / "checkpoints"
    plot_dir: Path = outputs_dir / "plots"
    sample_dir: Path = outputs_dir / "samples"
    checkpoint_path: Path = checkpoint_dir / "best_transformer_translation.pt"
    loss_curve_path: Path = plot_dir / "loss_curve.png"
    sample_output_path: Path = sample_dir / "translation_samples.txt"

    src_lang: str = "en"
    tgt_lang: str = "de"
    lowercase: bool = True
    min_freq: int = 1
    max_len: int = 128

    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_q_heads: int = 8
    num_kv_heads: int = 4
    attention_type: str = "gqa"
    d_ff: int = 1024
    dropout: float = 0.1
    ffn_activation: str = "gelu"
    rope_base: int = 10000

    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    clip_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_decode_len: int = 50
    sample_count: int = 5

    def ensure_dirs(self) -> None:
        for directory in (
            self.processed_dir,
            self.checkpoint_dir,
            self.plot_dir,
            self.sample_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    config = Config()
    config.ensure_dirs()
    return config
