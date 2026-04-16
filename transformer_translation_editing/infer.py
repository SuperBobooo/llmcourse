import os
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from configs import Config, get_config
from utils.editing import load_checkpoint_bundle, translate_sentence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy decoding for Transformer MT")
    parser.add_argument(
        "--sentence",
        type=str,
        default="i like apples",
        help="Source sentence to translate.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional checkpoint path override. Useful for edited checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else config.checkpoint_path
    model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, config, _ = load_checkpoint_bundle(
        checkpoint_path=checkpoint_path,
        config=config,
    )
    translation = translate_sentence(
        sentence=args.sentence,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=config,
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Source    : {args.sentence}")
    print(f"Generated : {translation}")


if __name__ == "__main__":
    main()
