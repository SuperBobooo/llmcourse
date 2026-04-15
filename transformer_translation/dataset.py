from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils.tokenizer import BasicTokenizer
from utils.vocab import Vocab


def read_parallel_corpus(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            src_text, tgt_text = parts
            pairs.append((src_text, tgt_text))
    return pairs


class TranslationDataset(Dataset):
    """
    Each example returns:
        src_ids: source tokens + <eos>
        tgt_input_ids: <bos> + target tokens
        tgt_output_ids: target tokens + <eos>
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_tokenizer: BasicTokenizer,
        tgt_tokenizer: BasicTokenizer,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int,
    ) -> None:
        self.pairs = list(pairs)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def _truncate(self, tokens: List[str], reserved_tokens: int) -> List[str]:
        return tokens[: self.max_len - reserved_tokens]

    def __getitem__(self, index: int) -> dict:
        src_text, tgt_text = self.pairs[index]

        src_tokens = self._truncate(self.src_tokenizer.tokenize(src_text), reserved_tokens=1)
        tgt_tokens = self._truncate(self.tgt_tokenizer.tokenize(tgt_text), reserved_tokens=1)

        src_ids = self.src_vocab.encode(src_tokens, add_eos=True)
        tgt_input_ids = self.tgt_vocab.encode(tgt_tokens, add_bos=True)
        tgt_output_ids = self.tgt_vocab.encode(tgt_tokens, add_eos=True)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids, dtype=torch.long),
            "tgt_output_ids": torch.tensor(tgt_output_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def build_collate_fn(pad_id: int) -> Callable[[List[dict]], dict]:
    def collate_fn(batch: List[dict]) -> dict:
        batch_size = len(batch)
        src_max_len = max(item["src_ids"].size(0) for item in batch)
        tgt_max_len = max(item["tgt_input_ids"].size(0) for item in batch)

        src_batch = torch.full((batch_size, src_max_len), pad_id, dtype=torch.long)
        tgt_input_batch = torch.full((batch_size, tgt_max_len), pad_id, dtype=torch.long)
        tgt_output_batch = torch.full((batch_size, tgt_max_len), pad_id, dtype=torch.long)

        src_texts = []
        tgt_texts = []

        for idx, item in enumerate(batch):
            src_len = item["src_ids"].size(0)
            tgt_len = item["tgt_input_ids"].size(0)

            src_batch[idx, :src_len] = item["src_ids"]
            tgt_input_batch[idx, :tgt_len] = item["tgt_input_ids"]
            tgt_output_batch[idx, :tgt_len] = item["tgt_output_ids"]

            src_texts.append(item["src_text"])
            tgt_texts.append(item["tgt_text"])

        return {
            "src_ids": src_batch,
            "tgt_input_ids": tgt_input_batch,
            "tgt_output_ids": tgt_output_batch,
            "src_texts": src_texts,
            "tgt_texts": tgt_texts,
        }

    return collate_fn

