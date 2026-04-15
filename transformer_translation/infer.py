import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from configs import Config, get_config
from models import TransformerTranslationModel
from utils import BasicTokenizer, Vocab, make_padding_mask


MODEL_CONFIG_KEYS = {
    "d_model",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_q_heads",
    "num_kv_heads",
    "attention_type",
    "d_ff",
    "dropout",
    "ffn_activation",
    "rope_base",
    "lowercase",
    "max_src_len",
    "max_tgt_len",
    "max_decode_len",
}


def apply_saved_model_config(config: Config, saved_config: dict | None) -> Config:
    if not saved_config:
        return config

    for key, value in saved_config.items():
        if key in MODEL_CONFIG_KEYS and hasattr(config, key):
            setattr(config, key, value)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def load_trained_model(
    config: Config | None = None,
) -> tuple[
    TransformerTranslationModel,
    BasicTokenizer,
    BasicTokenizer,
    Vocab,
    Vocab,
    Config,
]:
    config = config or get_config()
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.checkpoint_path}. Please run train.py first."
        )
    if not config.src_vocab_path.exists() or not config.tgt_vocab_path.exists():
        raise FileNotFoundError("Vocabulary files are missing. Please run train.py first.")

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    config = apply_saved_model_config(config, checkpoint.get("config"))

    src_vocab = Vocab.load(config.src_vocab_path)
    tgt_vocab = Vocab.load(config.tgt_vocab_path)

    model = TransformerTranslationModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    src_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    tgt_tokenizer = BasicTokenizer(lowercase=config.lowercase)
    return model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, config


def decode_generated_ids(
    generated_ids: list[int], tgt_vocab: Vocab, tgt_tokenizer: BasicTokenizer
) -> str:
    tokens = []
    for token_id in generated_ids:
        token = tgt_vocab.id_to_token(token_id)
        if token == Vocab.BOS_TOKEN:
            continue
        if token in {Vocab.EOS_TOKEN, Vocab.PAD_TOKEN}:
            break
        tokens.append(token)
    return tgt_tokenizer.detokenize(tokens)


@torch.no_grad()
def translate_sentence(
    sentence: str,
    model: TransformerTranslationModel,
    src_tokenizer: BasicTokenizer,
    tgt_tokenizer: BasicTokenizer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    config: Config,
) -> str:
    src_tokens = src_tokenizer.tokenize(sentence)[: config.max_src_len - 1]
    src_ids = torch.tensor(
        src_vocab.encode(src_tokens, add_eos=True),
        dtype=torch.long,
        device=config.device,
    ).unsqueeze(0)
    src_mask = make_padding_mask(src_ids, pad_id=src_vocab.pad_id)

    generated_ids = model.greedy_decode(
        src_ids=src_ids,
        src_mask=src_mask,
        bos_id=tgt_vocab.bos_id,
        eos_id=tgt_vocab.eos_id,
        max_len=config.max_decode_len,
    )[0].tolist()
    return decode_generated_ids(generated_ids, tgt_vocab, tgt_tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy decoding for Transformer MT")
    parser.add_argument(
        "--sentence",
        type=str,
        default="i like apples",
        help="Source sentence to translate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, config = load_trained_model()
    translation = translate_sentence(
        sentence=args.sentence,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=config,
    )
    print(f"Source   : {args.sentence}")
    print(f"Generated: {translation}")


if __name__ == "__main__":
    main()
