import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from configs import get_config, get_edit_config
from dataset import read_parallel_corpus
from utils.edit_data import (
    build_batch_from_pairs,
    ensure_text_known,
    find_matching_target,
    select_retain_pairs,
)
from utils.editing import (
    assess_edit_case,
    build_eval_loader,
    evaluate_model_on_loader,
    load_checkpoint_bundle,
    save_json,
    translate_sentence,
)
from utils.masks import (
    make_cross_attention_mask,
    make_decoder_self_attention_mask,
    make_padding_mask,
)


def parse_args() -> argparse.Namespace:
    default_config = get_edit_config()
    parser = argparse.ArgumentParser(description="Local knowledge editing for translation")
    parser.add_argument("--edit_source_text", type=str, default=default_config.edit_source_text)
    parser.add_argument("--edit_target_text", type=str, default=default_config.edit_target_text)
    parser.add_argument("--retain_set_size", type=int, default=default_config.retain_set_size)
    parser.add_argument("--edit_steps", type=int, default=default_config.edit_steps)
    parser.add_argument("--edit_lr", type=float, default=default_config.edit_lr)
    parser.add_argument(
        "--locality_loss_weight",
        type=float,
        default=default_config.locality_loss_weight,
    )
    parser.add_argument(
        "--edit_scope",
        type=str,
        default=default_config.edit_scope,
        choices=["lm_head", "decoder_last_ffn", "decoder_last_proj"],
    )
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
    parser.add_argument(
        "--edit_clip_grad_norm",
        type=float,
        default=default_config.edit_clip_grad_norm,
    )
    return parser.parse_args()


def apply_cli_to_edit_config(args: argparse.Namespace):
    edit_config = get_edit_config()
    edit_config.edit_source_text = args.edit_source_text
    edit_config.edit_target_text = args.edit_target_text
    edit_config.retain_set_size = args.retain_set_size
    edit_config.edit_steps = args.edit_steps
    edit_config.edit_lr = args.edit_lr
    edit_config.locality_loss_weight = args.locality_loss_weight
    edit_config.edit_scope = args.edit_scope
    edit_config.base_checkpoint_path = Path(args.base_checkpoint_path)
    edit_config.edited_checkpoint_path = Path(args.edited_checkpoint_path)
    edit_config.edit_clip_grad_norm = args.edit_clip_grad_norm
    edit_config.ensure_dirs()
    return edit_config


def select_editable_parameters(model, edit_scope: str) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    selected_parameters: list[tuple[str, torch.nn.Parameter]] = []

    if edit_scope == "lm_head":
        modules = [("output_projection", model.output_projection)]
    elif edit_scope == "decoder_last_ffn":
        modules = [("decoder.layers[-1].ffn", model.decoder.layers[-1].ffn)]
    elif edit_scope == "decoder_last_proj":
        modules = [
            ("decoder.layers[-1].self_attn.out_proj", model.decoder.layers[-1].self_attn.out_proj),
            (
                "decoder.layers[-1].cross_attn.out_proj",
                model.decoder.layers[-1].cross_attn.out_proj,
            ),
        ]
    else:
        raise ValueError(f"Unsupported edit_scope: {edit_scope}")

    for module_name, module in modules:
        for parameter_name, parameter in module.named_parameters():
            parameter.requires_grad = True
            selected_parameters.append((f"{module_name}.{parameter_name}", parameter))

    if not selected_parameters:
        raise RuntimeError(f"No parameters selected for edit_scope={edit_scope}")
    return selected_parameters


def compute_teacher_forcing_logits(model, batch: dict, src_pad_id: int, tgt_pad_id: int) -> torch.Tensor:
    src_ids = batch["src_ids"]
    tgt_input_ids = batch["tgt_input_ids"]
    src_mask = make_padding_mask(src_ids, pad_id=src_pad_id)
    tgt_self_attn_mask = make_decoder_self_attention_mask(tgt_input_ids, pad_id=tgt_pad_id)
    cross_attn_mask = make_cross_attention_mask(tgt_input_ids, src_ids, pad_id=src_pad_id)
    return model(
        src_ids=src_ids,
        tgt_input_ids=tgt_input_ids,
        src_mask=src_mask,
        tgt_self_attn_mask=tgt_self_attn_mask,
        cross_attn_mask=cross_attn_mask,
    )


def compute_edit_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=pad_id,
    )


def compute_locality_kl(
    edited_logits: torch.Tensor,
    base_logits: torch.Tensor,
    temperature: float,
    targets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    edited_log_probs = F.log_softmax(edited_logits / temperature, dim=-1)
    base_probs = F.softmax(base_logits / temperature, dim=-1)
    token_kl = F.kl_div(edited_log_probs, base_probs, reduction="none").sum(dim=-1)
    valid_mask = targets.ne(pad_id).float()
    masked_kl = token_kl * valid_mask
    normalizer = valid_mask.sum().clamp_min(1.0)
    return masked_kl.sum() / normalizer * (temperature ** 2)


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    edit_config = apply_cli_to_edit_config(args)
    base_config = get_config()

    (
        model,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        base_config,
        base_checkpoint,
    ) = load_checkpoint_bundle(edit_config.base_checkpoint_path, config=base_config)

    ensure_text_known(edit_config.edit_source_text, src_tokenizer, src_vocab, side="Source")
    ensure_text_known(edit_config.edit_target_text, tgt_tokenizer, tgt_vocab, side="Target")

    train_pairs = read_parallel_corpus(base_config.train_path)
    val_pairs = read_parallel_corpus(base_config.val_path)
    all_pairs = train_pairs + val_pairs
    original_target_text = find_matching_target(
        pairs=all_pairs,
        source_text=edit_config.edit_source_text,
        lowercase=base_config.lowercase,
    )

    edit_batch = build_batch_from_pairs(
        pairs=[(edit_config.edit_source_text, edit_config.edit_target_text)],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
        device=base_config.device,
    )
    retain_pairs = select_retain_pairs(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        edit_source_text=edit_config.edit_source_text,
        retain_set_size=edit_config.retain_set_size,
        lowercase=base_config.lowercase,
        seed=edit_config.seed,
    )
    retain_batch = build_batch_from_pairs(
        pairs=retain_pairs,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
        device=base_config.device,
    )

    with torch.no_grad():
        before_translation = translate_sentence(
            sentence=edit_config.edit_source_text,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=base_config,
        )
        base_retain_logits = compute_teacher_forcing_logits(
            model=model,
            batch=retain_batch,
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
        ).detach()
        val_loader = build_eval_loader(
            config=base_config,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
        )
        before_val_metrics = evaluate_model_on_loader(
            model=model,
            data_loader=val_loader,
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
            config=base_config,
        )
        case_assessment = assess_edit_case(before_translation)

    editable_parameters = select_editable_parameters(model, edit_scope=edit_config.edit_scope)
    optimizer = AdamW(
        [parameter for _, parameter in editable_parameters],
        lr=edit_config.edit_lr,
        weight_decay=0.0,
    )

    print(f"Base checkpoint : {edit_config.base_checkpoint_path}")
    print(f"Edit scope      : {edit_config.edit_scope}")
    print(f"Edit request    : {edit_config.edit_source_text!r} -> {edit_config.edit_target_text!r}")
    if original_target_text is not None:
        print(f"Original target : {original_target_text!r}")
    print(f"Case type       : {case_assessment['case_type']}")
    print(f"Case note       : {case_assessment['note']}")
    print("Trainable params:")
    for parameter_name, _ in editable_parameters:
        print(f"  - {parameter_name}")

    edit_history = []
    best_state = clone_model_state(model)
    best_rank = (1, float("inf"), float("inf"))
    best_step = 0
    best_translation = before_translation
    model.train()
    for step in range(1, edit_config.edit_steps + 1):
        optimizer.zero_grad()

        edit_logits = compute_teacher_forcing_logits(
            model=model,
            batch=edit_batch,
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
        )
        edit_loss = compute_edit_ce_loss(
            logits=edit_logits,
            targets=edit_batch["tgt_output_ids"],
            pad_id=tgt_vocab.pad_id,
        )

        retain_logits = compute_teacher_forcing_logits(
            model=model,
            batch=retain_batch,
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
        )
        locality_loss = compute_locality_kl(
            edited_logits=retain_logits,
            base_logits=base_retain_logits,
            temperature=edit_config.kl_temperature,
            targets=retain_batch["tgt_output_ids"],
            pad_id=tgt_vocab.pad_id,
        )

        total_loss = edit_loss + edit_config.locality_loss_weight * locality_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in editable_parameters],
            max_norm=edit_config.edit_clip_grad_norm,
        )
        optimizer.step()

        with torch.no_grad():
            model.eval()
            current_translation = translate_sentence(
                sentence=edit_config.edit_source_text,
                model=model,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                config=base_config,
            )
            current_success = current_translation == edit_config.edit_target_text
            current_rank = (
                0 if current_success else 1,
                locality_loss.item() if current_success else edit_loss.item(),
                total_loss.item(),
            )
            if current_rank < best_rank:
                best_rank = current_rank
                best_state = clone_model_state(model)
                best_step = step
                best_translation = current_translation
            model.train()

        record = {
            "step": step,
            "edit_loss": round(edit_loss.item(), 6),
            "locality_loss": round(locality_loss.item(), 6),
            "total_loss": round(total_loss.item(), 6),
            "current_translation": current_translation,
            "current_success": current_success,
        }
        edit_history.append(record)
        if step == 1 or step % 10 == 0 or step == edit_config.edit_steps:
            print(
                f"Step {step:03d}/{edit_config.edit_steps} | "
                f"edit_loss={edit_loss.item():.4f} | "
                f"locality_loss={locality_loss.item():.4f} | "
                f"total_loss={total_loss.item():.4f} | "
                f"translation={current_translation!r}"
            )

    model.load_state_dict(best_state)
    model.eval()
    after_translation = translate_sentence(
        sentence=edit_config.edit_source_text,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=base_config,
    )
    after_val_metrics = evaluate_model_on_loader(
        model=model,
        data_loader=val_loader,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=base_config,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": base_checkpoint.get("config", asdict(base_config)),
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "base_checkpoint_path": str(edit_config.base_checkpoint_path),
            "edit_config": asdict(edit_config),
            "before_translation": before_translation,
            "after_translation": after_translation,
        },
        edit_config.edited_checkpoint_path,
    )

    summary = {
        "edit_request": {
            "source_text": edit_config.edit_source_text,
            "target_text": edit_config.edit_target_text,
            "edit_scope": edit_config.edit_scope,
            "original_target_text": original_target_text,
        },
        "checkpoint_paths": {
            "base_checkpoint_path": str(edit_config.base_checkpoint_path),
            "edited_checkpoint_path": str(edit_config.edited_checkpoint_path),
        },
        "case_assessment": case_assessment,
        "selected_edit_step": best_step,
        "before_translation": before_translation,
        "best_step_translation": best_translation,
        "after_translation": after_translation,
        "before_val_metrics": before_val_metrics,
        "after_val_metrics": after_val_metrics,
        "retain_pairs": retain_pairs,
        "edit_history": edit_history,
    }
    save_json(summary, edit_config.edit_summary_path)

    print("\nEdit summary")
    print(f"  Before: {before_translation}")
    print(f"  After : {after_translation}")
    print(f"  Best edit step       : {best_step}")
    print(f"  Edited checkpoint saved to: {edit_config.edited_checkpoint_path}")
    print(f"  Report saved to          : {edit_config.edit_summary_path}")


if __name__ == "__main__":
    main()
