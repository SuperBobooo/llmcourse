import re
from typing import List


class BasicTokenizer:
    """A lightweight whitespace + punctuation tokenizer for teaching demos."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def detokenize(self, tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

