# src/utils/fingerprint.py
from typing import Any, Iterable, Optional
import hashlib
import json


def stable_hash(obj: Any) -> str:
    """Return a deterministic SHA256 hash for JSON-serializable objects."""

    def _default(value: Any):
        return str(value)

    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_texts(texts: Iterable[str], limit: Optional[int] = None) -> str:
    """Hash a sequence of strings in order to fingerprint row-level text inputs."""
    h = hashlib.sha256()
    count = 0
    for item in texts:
        if limit is not None and count >= limit:
            break
        h.update(str(item).encode("utf-8", errors="ignore"))
        h.update(b"\n")
        count += 1
    h.update(f"count={count}".encode("utf-8"))
    return h.hexdigest()


def hash_array_signature(shape: Any, dtype: Any, stats: Optional[dict] = None) -> str:
    """Hash a lightweight array signature without materializing full content."""
    payload = {
        "shape": tuple(shape) if shape is not None else None,
        "dtype": str(dtype),
        "stats": stats or {},
    }
    return stable_hash(payload)
