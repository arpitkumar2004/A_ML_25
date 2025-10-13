"""Text cleaning utilities."""

import re
import pandas as pd
import numpy as np


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)  # strip html
    s = s.strip()
    return s

