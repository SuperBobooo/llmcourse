import argparse
from pathlib import Path

import torch

from configs import get_config, get_edit_config
from dataset import read_parallel_corpus
from edit_knowledge import compute_locality_kl, compute_teacher_forcing_logits
from utils.edit_data import build_batch_from_pairs, find_matching_target, select_retain_pairs
from utils.editing import (
    assess_edit_case,
    build_eval_loader,
    evaluate_model_on_loader,
    load_checkpoint_bundle,
    save_json,
    translate_sentence,
)


def parse_args() -> argparse.Namespace:
    default_config = get_edit_config()
    parser = argparse.ArgumentParser(description="Evaluate before/after knowledge editing")
    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        default=str(default_config.base_checkpoint_path),
    )
    parser.add_argument(
        "--edited_checkpoint_path",
        type=str,
        default=str(default_config.edited_checkpoint_path),
    )
    parser.add_argument("--edit_source_text", type=str, default=default_config.edit_source_text)
    parser.add_argument("--edit_target_text", type=str, default=default_config.edit_target_text)
    parser.add_argument("--retain_set_size", type=int, default=default_config.retain_set_size)
    return parser.parse_args()


@torch.no_grad()
def compare_locality(
    base_model,
    edited_model,
    retain_pairs,
    src_tokenizer,
    tgt_tokenizer,
    src_vocab,
    tgt_vocab,
    config,
) -> dict:
    unchanged = 0
    translations = []
    for src_text, tgt_text in retain_pairs:
        base_translation = translate_sentence(
            sentence=src_text,
            model=base_model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=config,
        )
        edited_translation = translate_sentence(
            sentence=src_text,
            model=edited_model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=config,
        )
        if base_translation == edited_translation:
            unchanged += 1
        translations.append(
            {
                "source": src_text,
                "reference": tgt_text,
                "before": base_translation,
                "after": edited_translation,
            }
        )

    retain_batch = build_batch_from_pairs(
        pairs=retain_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=config,
        device=config.device,
    )
    base_logits = compute_teacher_forcing_logits(
        model=base_model,
        batch=retain_batch,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    edited_logits = compute_teacher_forcing_logits(
        model=edited_model,
        batch=retain_batch,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    retain_kl = compute_locality_kl(
        edited_logits=edited_logits,
        base_logits=base_logits,
        temperature=1.0,
        targets=retain_batch["tgt_output_ids"],
        pad_id=tgt_vocab.pad_id,
    ).item()

    return {
        "unchanged_ratio": unchanged / max(1, len(retain_pairs)),
        "changed_ratio": 1.0 - (unchanged / max(1, len(retain_pairs))),
        "retain_kl": retain_kl,
        "translations": translations,
    }


def main() -> None:
    args = parse_args()
    edit_config = get_edit_config()
    edit_config.base_checkpoint_path = Path(args.base_checkpoint_path)
    edit_config.edited_checkpoint_path = Path(args.edited_checkpoint_path)
    edit_config.edit_source_text = args.edit_source_text
    edit_config.edit_target_text = args.edit_target_text
    edit_config.retain_set_size = args.retain_set_size
    edit_config.ensure_dirs()

    base_config = get_config()
    (
        base_model,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        base_config,
        _,
    ) = load_checkpoint_bundle(edit_config.base_checkpoint_path, config=base_config)
    (
        edited_model,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = load_checkpoint_bundle(edit_config.edited_checkpoint_path, config=base_config)

    before_translation = translate_sentence(
        sentence=edit_config.edit_source_text,
        model=base_model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
    )
    after_translation = translate_sentence(
        sentence=edit_config.edit_source_text,
        model=edited_model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
    )

    train_pairs = read_parallel_corpus(base_config.train_path)
    val_pairs = read_parallel_corpus(base_config.val_path)
    all_pairs = train_pairs + val_pairs
    original_target_text = find_matching_target(
        pairs=all_pairs,
        source_text=edit_config.edit_source_text,
        lowercase=base_config.lowercase,
    )
    retain_pairs = select_retain_pairs(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        edit_source_text=edit_config.edit_source_text,
        retain_set_size=edit_config.retain_set_size,
        lowercase=base_config.lowercase,
        seed=edit_config.seed,
    )
    locality_metrics = compare_locality(
        base_model=base_model,
        edited_model=edited_model,
        retain_pairs=retain_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
    )

    val_loader = build_eval_loader(
        config=base_config,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )
    before_val_metrics = evaluate_model_on_loader(
        model=base_model,
        data_loader=val_loader,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=base_config,
    )
    after_val_metrics = evaluate_model_on_loader(
        model=edited_model,
        data_loader=val_loader,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=base_config,
    )
    case_assessment = assess_edit_case(before_translation)

    summary = {
        "edit_request": {
            "source_text": edit_config.edit_source_text,
            "target_text": edit_config.edit_target_text,
            "original_target_text": original_target_text,
        },
        "edit_success": {
            "before_translation": before_translation,
            "after_translation": after_translation,
            "target_text": edit_config.edit_target_text,
            "success_before": before_translation == edit_config.edit_target_text,
            "success_after": after_translation == edit_config.edit_target_text,
        },
        "case_assessment": case_assessment,
        "locality": locality_metrics,
        "general_performance": {
            "before_val": before_val_metrics,
            "after_val": after_val_metrics,
            "delta_loss": after_val_metrics["loss"] - before_val_metrics["loss"],
            "delta_accuracy": after_val_metrics["accuracy"] - before_val_metrics["accuracy"],
        },
    }
    save_json(summary, edit_config.comparison_summary_path)

    print("Edit Success")
    print(f"  Before: {before_translation}")
    print(f"  After : {after_translation}")
    print(f"  Target: {edit_config.edit_target_text}")
    if original_target_text is not None:
        print(f"  Original target in TSV: {original_target_text}")
    print(f"  Existing-knowledge case: {case_assessment['is_existing_knowledge_case']}")
    print(f"  Case note             : {case_assessment['note']}")
    print(f"  Success before: {summary['edit_success']['success_before']}")
    print(f"  Success after : {summary['edit_success']['success_after']}")

    print("\nLocality")
    print(f"  Unchanged translation ratio: {locality_metrics['unchanged_ratio']:.4f}")
    print(f"  Changed translation ratio  : {locality_metrics['changed_ratio']:.4f}")
    print(f"  Retain KL               : {locality_metrics['retain_kl']:.6f}")

    print("\nGeneral Performance")
    print(f"  Base val loss   : {before_val_metrics['loss']:.4f}")
    print(f"  Edited val loss : {after_val_metrics['loss']:.4f}")
    print(f"  Base val acc    : {before_val_metrics['accuracy']:.4f}")
    print(f"  Edited val acc  : {after_val_metrics['accuracy']:.4f}")
    print(
        f"  Delta loss      : "
        f"{summary['general_performance']['delta_loss']:+.4f}"
    )
    print(
        f"  Delta acc       : "
        f"{summary['general_performance']['delta_accuracy']:+.4f}"
    )
    print(f"  Report saved to : {edit_config.comparison_summary_path}")


if __name__ == "__main__":
    main()
