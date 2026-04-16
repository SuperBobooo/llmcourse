import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = project_root / "data" / "raw"
    train_path: Path = raw_dir / "train.tsv"
    val_path: Path = raw_dir / "val.tsv"
    test_path: Path = raw_dir / "test.tsv"
    processed_dir: Path = project_root / "data" / "processed"
    src_vocab_path: Path = processed_dir / "src_vocab.json"
    tgt_vocab_path: Path = processed_dir / "tgt_vocab.json"
    dataset_metadata_path: Path = processed_dir / "dataset_meta.json"
    outputs_dir: Path = project_root / "outputs"
    checkpoint_dir: Path = outputs_dir / "checkpoints"
    plot_dir: Path = outputs_dir / "plots"
    sample_dir: Path = outputs_dir / "samples"
    checkpoint_path: Path = checkpoint_dir / "best_transformer_translation.pt"
    loss_curve_path: Path = plot_dir / "loss_curve.png"
    sample_output_path: Path = sample_dir / "translation_samples.txt"

    src_lang: str = "en"
    tgt_lang: str = "de"
    dataset_name: str = "iwslt2017_en_de"
    max_train_samples: int = 80000
    max_val_samples: int = 2000
    max_test_samples: int = 2000
    lowercase: bool = True
    # Keep all content words on the toy corpus so the editing demo can
    # modify an existing translation fact instead of fighting <unk>.
    min_freq: int = 1
    max_vocab_size: int = 32000
    max_src_len: int = 80
    max_tgt_len: int = 80

    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_q_heads: int = 8
    num_kv_heads: int = 4
    attention_type: str = "gqa"
    d_ff: int = 1024
    dropout: float = 0.0
    ffn_activation: str = "gelu"
    rope_base: int = 10000

    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 40
    clip_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    early_stopping_patience: int = 10

    num_workers: int = 4 if os.name != "nt" else 0
    pin_memory: bool = torch.cuda.is_available()
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_decode_len: int = 80
    sample_count: int = 5

    @property
    def max_len(self) -> int:
        return max(self.max_src_len, self.max_tgt_len)

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
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
