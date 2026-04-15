import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from .presets import PRESET_REGISTRY


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = field(init=False)
    train_path: Path = field(init=False)
    val_path: Path = field(init=False)
    test_path: Path = field(init=False)
    processed_dir: Path = field(init=False)
    dataset_metadata_path: Path = field(init=False)
    outputs_root: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    plot_dir: Path = field(init=False)
    sample_dir: Path = field(init=False)
    src_vocab_path: Path = field(init=False)
    tgt_vocab_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    loss_curve_path: Path = field(init=False)
    sample_output_path: Path = field(init=False)
    results_table_path: Path = field(init=False)

    preset_name: str = "baseline"
    experiment_name: str = "baseline_gqa_dense"

    src_lang: str = "en"
    tgt_lang: str = "de"
    dataset_name: str = "iwslt2017_en_de"
    max_train_samples: int = 80000
    max_val_samples: int = 2000
    max_test_samples: int = 2000
    lowercase: bool = True
    min_freq: int = 2
    max_vocab_size: int = 32000
    max_src_len: int = 80
    max_tgt_len: int = 80

    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 4
    attention_type: str = "gqa"
    d_ff: int = 1024
    dropout: float = 0.1
    ffn_activation: str = "gelu"
    rope_base: int = 10000

    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    expert_hidden_dim: int | None = None
    use_moe_aux_loss: bool = True
    moe_aux_loss_coef: float = 1e-2

    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    clip_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    early_stopping_patience: int = 2

    num_workers: int = 4 if os.name != "nt" else 0
    pin_memory: bool = torch.cuda.is_available()
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_decode_len: int = 80
    sample_count: int = 5

    @property
    def num_q_heads(self) -> int:
        return self.num_heads

    @property
    def ffn_type(self) -> str:
        return "moe_ffn" if self.use_moe else "dense_ffn"

    @property
    def max_len(self) -> int:
        return max(self.max_src_len, self.max_tgt_len)

    def finalize(self) -> None:
        self.raw_dir = self.project_root / "data" / "raw"
        self.train_path = self.raw_dir / "train.tsv"
        self.val_path = self.raw_dir / "val.tsv"
        self.test_path = self.raw_dir / "test.tsv"
        self.processed_dir = self.project_root / "data" / "processed"
        self.dataset_metadata_path = self.processed_dir / "dataset_meta.json"

        self.outputs_root = self.project_root / "outputs"
        self.checkpoint_dir = self.outputs_root / "checkpoints"
        self.plot_dir = self.outputs_root / "plots"
        self.sample_dir = self.outputs_root / "samples"
        self.results_table_path = self.outputs_root / "experiment_results.csv"

        attention_type = self.attention_type.lower()
        if attention_type == "mha":
            self.num_kv_heads = self.num_heads
        elif attention_type == "mqa":
            self.num_kv_heads = 1
        elif attention_type == "gqa":
            if self.num_heads < self.num_kv_heads:
                raise ValueError("GQA requires num_heads >= num_kv_heads.")
        else:
            raise ValueError("attention_type must be one of: mha, gqa, mqa")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if self.top_k_experts > self.num_experts:
            raise ValueError("top_k_experts must be <= num_experts.")
        if self.expert_hidden_dim is None:
            self.expert_hidden_dim = self.d_ff

        if not self.experiment_name:
            self.experiment_name = f"{self.attention_type}_{self.ffn_type}"

        self.src_vocab_path = self.processed_dir / f"{self.experiment_name}_src_vocab.json"
        self.tgt_vocab_path = self.processed_dir / f"{self.experiment_name}_tgt_vocab.json"
        self.checkpoint_path = (
            self.checkpoint_dir / f"{self.experiment_name}_best_transformer_translation.pt"
        )
        self.loss_curve_path = self.plot_dir / f"{self.experiment_name}_loss_curve.png"
        self.sample_output_path = (
            self.sample_dir / f"{self.experiment_name}_translation_samples.txt"
        )

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.checkpoint_dir,
            self.plot_dir,
            self.sample_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_config(preset_name: str | None = None) -> Config:
    config = Config()
    selected_preset = preset_name or config.preset_name
    if selected_preset not in PRESET_REGISTRY:
        raise ValueError(
            f"Unknown preset '{selected_preset}'. "
            f"Available presets: {', '.join(sorted(PRESET_REGISTRY))}"
        )

    for key, value in PRESET_REGISTRY[selected_preset].items():
        setattr(config, key, value)

    config.preset_name = selected_preset
    config.finalize()
    config.ensure_dirs()
    return config
