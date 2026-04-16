from dataclasses import dataclass
from pathlib import Path


@dataclass
class EditConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    outputs_dir: Path = project_root / "outputs"
    checkpoint_dir: Path = outputs_dir / "checkpoints"
    editing_dir: Path = outputs_dir / "editing"
    reports_dir: Path = outputs_dir / "reports"

    # The default edit is now a complete sentence that already exists in the
    # toy corpus, so the experiment better matches "modify existing knowledge".
    edit_source_text: str = "i like apples"
    edit_target_text: str = "ich mag bananen"
    retain_set_size: int = 12
    edit_steps: int = 40
    edit_lr: float = 1e-3
    locality_loss_weight: float = 6.0
    edit_scope: str = "lm_head"
    kl_temperature: float = 1.0
    edit_clip_grad_norm: float = 1.0
    seed: int = 42

    base_checkpoint_path: Path = checkpoint_dir / "best_transformer_translation.pt"
    edited_checkpoint_path: Path = editing_dir / "edited_transformer_translation.pt"
    edit_summary_path: Path = reports_dir / "edit_summary.json"
    comparison_summary_path: Path = reports_dir / "edit_comparison.json"

    def ensure_dirs(self) -> None:
        for directory in (self.editing_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_edit_config() -> EditConfig:
    config = EditConfig()
    config.ensure_dirs()
    return config
