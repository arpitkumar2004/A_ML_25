# src/data/augmentations.py
from typing import Any

# Placeholder augmentation module.
# For images, prefer albumentations or torchvision transforms in actual impl.

class Augmentations:
    @staticmethod
    def noop(item: Any) -> Any:
        """No-op augmentation (useful for pipeline tests)."""
        return item

    @staticmethod
    def example_text_noise(text: str, drop_prob: float = 0.01) -> str:
        """
        Simple text augmentation: randomly drop characters (not words).
        Only for experimentation — use sparingly.
        """
        import random
        if not text:
            return ""
        out = []
        for ch in text:
            if random.random() < drop_prob:
                continue
            out.append(ch)
        return "".join(out)
