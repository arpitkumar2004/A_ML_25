from typing import Optional, Iterable, List
import os
import numpy as np
import joblib

from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory
from ..utils.fingerprint import stable_hash

logger = LoggerFactory.get("image_embeddings")


class ImageEmbedder:
    """Extract image embeddings with fingerprinted cache reuse."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_path: Optional[str] = "data/processed/image_embeddings.joblib",
        device: Optional[str] = None,
        batch_size: int = 32,
        fallback_dim: int = 512,
        lazy_init: bool = True,
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.fallback_dim = fallback_dim
        self.device = device
        self.lazy_init = lazy_init

        self._backend = None
        self._model = None
        self._processor = None
        self._preprocess = None
        self._initialized = False

        if not self.lazy_init:
            self._init_model_if_possible()

    def _init_model_if_possible(self):
        if self._initialized:
            return
        try:
            import torch

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                import clip

                model, preprocess = clip.load("ViT-B/32", device=self.device)
                model.eval()
                self._backend = "openai"
                self._model = model
                self._preprocess = preprocess
                logger.info("Loaded OpenAI CLIP backend.")
                self._initialized = True
                return
            except Exception:
                pass

            try:
                from transformers import CLIPModel, CLIPProcessor

                model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model.eval()
                self._backend = "hf"
                self._model = model
                self._processor = processor
                logger.info(f"Loaded HF CLIP backend: {self.model_name}")
                self._initialized = True
                return
            except Exception as exc:
                logger.warning(f"HF CLIP unavailable, using zero-image fallback. reason={exc}")

        except Exception as exc:
            logger.warning(f"Torch/vision stack unavailable, using zero-image fallback. reason={exc}")

        self._backend = None
        self._model = None
        self._processor = None
        self._initialized = True

    def _ensure_initialized(self):
        if not self._initialized:
            self._init_model_if_possible()

    def _load_cache(self, fingerprint: Optional[str] = None):
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached image embeddings from {self.cache_path}")
            payload = joblib.load(self.cache_path)
            if isinstance(payload, dict) and "data" in payload and "fingerprint" in payload:
                if fingerprint is not None and payload.get("fingerprint") != fingerprint:
                    return None
                return payload.get("data")
            if fingerprint is not None:
                return None
            return payload
        return None

    def _save_cache(self, obj, fingerprint: Optional[str] = None):
        if not self.cache_path:
            return
        IO.save_pickle({"fingerprint": fingerprint, "data": obj}, self.cache_path)
        logger.info(f"Saved image embeddings to {self.cache_path}")

    def _fingerprint(self, image_paths: Iterable[str], mode: str) -> str:
        paths = [str(p) for p in image_paths]
        payload = {
            "mode": mode,
            "model_name": self.model_name,
            "n": len(paths),
            "paths_hash": stable_hash(paths),
            "backend": self._backend,
        }
        return stable_hash(payload)

    def _load_image(self, path_or_url: str):
        from PIL import Image

        try:
            if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
                import requests
                from io import BytesIO

                r = requests.get(path_or_url, timeout=10)
                return Image.open(BytesIO(r.content)).convert("RGB")

            if path_or_url and os.path.exists(path_or_url):
                return Image.open(path_or_url).convert("RGB")

        except Exception as exc:
            logger.warning(f"Failed to load image '{path_or_url}': {exc}")

        return Image.new("RGB", (224, 224), color=(255, 255, 255))

    def embed(self, image_paths: Iterable[str], use_cache: bool = True, fingerprint: Optional[str] = None):
        paths = list(image_paths)
        self._ensure_initialized()
        fp = fingerprint or self._fingerprint(paths, mode="embed")
        cached = self._load_cache(fp) if use_cache else None
        if cached is not None:
            return cached

        n = len(paths)
        if self._backend is None or self._model is None:
            emb = np.zeros((n, self.fallback_dim), dtype=np.float32)
            self._save_cache(emb, fp)
            return emb

        import torch

        outputs: List[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            imgs = [self._load_image(p) for p in batch_paths]

            if self._backend == "openai":
                x = torch.stack([self._preprocess(img).to(self.device) for img in imgs])
                with torch.no_grad():
                    y = self._model.encode_image(x).cpu().numpy()
            else:
                inputs = self._processor(images=imgs, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    raw = self._model.get_image_features(**inputs)
                    if hasattr(raw, "cpu"):
                        y = raw.cpu().numpy()
                    elif hasattr(raw, "image_embeds") and hasattr(raw.image_embeds, "cpu"):
                        y = raw.image_embeds.cpu().numpy()
                    elif hasattr(raw, "pooler_output") and hasattr(raw.pooler_output, "cpu"):
                        y = raw.pooler_output.cpu().numpy()
                    elif isinstance(raw, (tuple, list)) and len(raw) > 0 and hasattr(raw[0], "cpu"):
                        y = raw[0].cpu().numpy()
                    else:
                        raise TypeError(f"Unsupported HF CLIP output type: {type(raw)}")

            norm = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
            outputs.append(y / norm)

        emb = np.vstack(outputs) if outputs else np.zeros((n, self.fallback_dim), dtype=np.float32)
        self._save_cache(emb, fp)
        return emb
