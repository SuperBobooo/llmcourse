import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


class Vocab:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, tokens: Iterable[str]) -> None:
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @classmethod
    def build(cls, tokenized_texts: Iterable[List[str]], min_freq: int = 1) -> "Vocab":
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        specials = [cls.PAD_TOKEN, cls.UNK_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN]
        vocab_tokens = list(specials)
        for token, freq in sorted(counter.items()):
            if freq >= min_freq and token not in specials:
                vocab_tokens.append(token)
        return cls(vocab_tokens)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(
        self,
        tokens: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        token_ids = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self.token_to_id(token) for token in tokens)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        tokens = []
        specials = {
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        }
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return tokens

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump({"itos": self.itos}, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(payload["itos"])

