from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import torch


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    predictions = logits.argmax(dim=-1)
    valid_mask = targets.ne(pad_id)
    if valid_mask.sum().item() == 0:
        return 0.0
    correct = predictions.eq(targets) & valid_mask
    return correct.sum().item() / valid_mask.sum().item()


def save_loss_curve(
    train_losses: List[float], val_losses: List[float], output_path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Translation Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_translation_samples(samples: Iterable[dict], output_path: Path) -> None:
    lines = []
    for index, sample in enumerate(samples, start=1):
        lines.append(f"Sample {index}")
        lines.append(f"  Source   : {sample['source']}")
        lines.append(f"  Reference: {sample['reference']}")
        lines.append(f"  Generated: {sample['generated']}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

