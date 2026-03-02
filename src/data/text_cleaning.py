# src/data/text_cleaning.py
import re
from typing import Optional


class TextCleaner:
    """
    Lightweight text cleaning utilities.
    Methods are static for easy pipeline integration.
    """
    @staticmethod
    def basic(text: Optional[str]) -> str:
        if text is None:
            return ""
        s = str(text)
        s = s.strip()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.lower()
        # remove HTML tags
        s = re.sub(r"<[^>]*>", " ", s)
        # remove extra punctuation except basic separators
        s = re.sub(r"[^0-9a-zA-Z\.\,\-\s\%\/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
