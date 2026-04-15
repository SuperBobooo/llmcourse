import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_config
from utils.tokenizer import BasicTokenizer


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    hf_repo: str
    hf_config_name: str | None
    src_lang: str
    tgt_lang: str
    trust_remote_code: bool = False


DATASET_SPECS = {
    "iwslt2017_en_de": DatasetSpec(
        dataset_name="iwslt2017_en_de",
        hf_repo="IWSLT/iwslt2017",
        hf_config_name="iwslt2017-en-de",
        src_lang="en",
        tgt_lang="de",
        trust_remote_code=True,
    ),
    "multi30k_en_de": DatasetSpec(
        dataset_name="multi30k_en_de",
        hf_repo="bentrevett/multi30k",
        hf_config_name=None,
        src_lang="en",
        tgt_lang="de",
        trust_remote_code=False,
    ),
}


def parse_args() -> argparse.Namespace:
    config = get_config()
    parser = argparse.ArgumentParser(
        description="Prepare IWSLT2017 or Multi30k English-German data as TSV files."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=config.dataset_name,
        choices=sorted(DATASET_SPECS.keys()),
        help="Which dataset to download and export.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=config.max_train_samples,
        help="Maximum number of filtered training samples to export. Use <= 0 for all.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=config.max_val_samples,
        help="Maximum number of filtered validation samples to export. Use <= 0 for all.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=config.max_test_samples,
        help="Maximum number of filtered test samples to export. Use <= 0 for all.",
    )
    parser.add_argument(
        "--max_src_len",
        type=int,
        default=config.max_src_len,
        help="Maximum source sequence length including the later <eos> token.",
    )
    parser.add_argument(
        "--max_tgt_len",
        type=int,
        default=config.max_tgt_len,
        help="Maximum target sequence length including the later <bos>/<eos> token.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=config.lowercase,
        help="Lowercase text before token counting. Matches the current tokenizer setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.seed,
        help="Random seed used when shuffling the training split before truncation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TSV files if they already exist.",
    )
    return parser.parse_args()


def require_datasets_package():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for formal dataset preparation. "
            "Please run: pip install -r requirements.txt"
        ) from exc
    return load_dataset


def normalize_limit(limit: int) -> int | None:
    return None if limit <= 0 else limit


def extract_pair(example: dict, spec: DatasetSpec) -> tuple[str, str]:
    if "translation" in example and isinstance(example["translation"], dict):
        translation = example["translation"]
        return translation[spec.src_lang], translation[spec.tgt_lang]

    if spec.src_lang in example and spec.tgt_lang in example:
        return example[spec.src_lang], example[spec.tgt_lang]

    raise KeyError(
        f"Cannot extract translation fields for {spec.dataset_name}. "
        f"Available keys: {sorted(example.keys())}"
    )


def keep_pair(
    src_text: str,
    tgt_text: str,
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    max_src_len: int,
    max_tgt_len: int,
) -> bool:
    src_text = src_text.strip()
    tgt_text = tgt_text.strip()
    if not src_text or not tgt_text:
        return False

    src_len = len(src_tokenizer.tokenize(src_text))
    tgt_len = len(tgt_tokenizer.tokenize(tgt_text))

    if src_len == 0 or tgt_len == 0:
        return False

    return src_len <= max_src_len - 1 and tgt_len <= max_tgt_len - 1


def load_split_dataset(load_dataset, spec: DatasetSpec, split: str):
    if spec.hf_config_name is None:
        return load_dataset(
            spec.hf_repo,
            split=split,
            trust_remote_code=spec.trust_remote_code,
        )
    return load_dataset(
        spec.hf_repo,
        spec.hf_config_name,
        split=split,
        trust_remote_code=spec.trust_remote_code,
    )


def collect_pairs(
    dataset_split,
    spec: DatasetSpec,
    split_name: str,
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    max_src_len: int,
    max_tgt_len: int,
    max_samples: int | None,
    seed: int,
) -> list[tuple[str, str]]:
    if split_name == "train":
        dataset_split = dataset_split.shuffle(seed=seed)

    collected = []
    skipped = 0

    for example in dataset_split:
        src_text, tgt_text = extract_pair(example, spec)
        if not keep_pair(
            src_text=src_text,
            tgt_text=tgt_text,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        ):
            skipped += 1
            continue

        collected.append((src_text.strip(), tgt_text.strip()))
        if max_samples is not None and len(collected) >= max_samples:
            break

    print(
        f"{split_name:<10} kept={len(collected):>6} | skipped={skipped:>6} | "
        f"limit={'all' if max_samples is None else max_samples}"
    )
    return collected


def write_tsv(path: Path, pairs: list[tuple[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Re-run with --overwrite to replace it."
        )

    with path.open("w", encoding="utf-8") as file:
        for src_text, tgt_text in pairs:
            file.write(f"{src_text}\t{tgt_text}\n")


def save_metadata(
    output_path: Path,
    spec: DatasetSpec,
    args: argparse.Namespace,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
) -> None:
    metadata = {
        "dataset_name": spec.dataset_name,
        "hf_repo": spec.hf_repo,
        "hf_config_name": spec.hf_config_name,
        "src_lang": spec.src_lang,
        "tgt_lang": spec.tgt_lang,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "test_samples": len(test_pairs),
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "lowercase": args.lowercase,
        "requested_max_train_samples": normalize_limit(args.max_train_samples),
        "requested_max_val_samples": normalize_limit(args.max_val_samples),
        "requested_max_test_samples": normalize_limit(args.max_test_samples),
        "seed": args.seed,
        "command_args": vars(args),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = get_config()
    config.ensure_dirs()

    spec = DATASET_SPECS[args.dataset_name]
    load_dataset = require_datasets_package()

    src_tokenizer = BasicTokenizer(lowercase=args.lowercase)
    tgt_tokenizer = BasicTokenizer(lowercase=args.lowercase)

    print(
        f"Preparing dataset={spec.dataset_name} from Hugging Face repo={spec.hf_repo}"
    )
    if spec.hf_config_name is not None:
        print(f"Using config name: {spec.hf_config_name}")

    train_split = load_split_dataset(load_dataset, spec, split="train")
    val_split = load_split_dataset(load_dataset, spec, split="validation")
    test_split = load_split_dataset(load_dataset, spec, split="test")

    train_pairs = collect_pairs(
        dataset_split=train_split,
        spec=spec,
        split_name="train",
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_samples=normalize_limit(args.max_train_samples),
        seed=args.seed,
    )
    val_pairs = collect_pairs(
        dataset_split=val_split,
        spec=spec,
        split_name="validation",
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_samples=normalize_limit(args.max_val_samples),
        seed=args.seed,
    )
    test_pairs = collect_pairs(
        dataset_split=test_split,
        spec=spec,
        split_name="test",
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_samples=normalize_limit(args.max_test_samples),
        seed=args.seed,
    )

    write_tsv(config.train_path, train_pairs, overwrite=args.overwrite)
    write_tsv(config.val_path, val_pairs, overwrite=args.overwrite)
    write_tsv(config.test_path, test_pairs, overwrite=args.overwrite)
    save_metadata(
        output_path=config.dataset_metadata_path,
        spec=spec,
        args=args,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )

    print("\nExport complete:")
    print(f"  train -> {config.train_path}")
    print(f"  val   -> {config.val_path}")
    print(f"  test  -> {config.test_path}")
    print(f"  meta  -> {config.dataset_metadata_path}")


if __name__ == "__main__":
    main()

