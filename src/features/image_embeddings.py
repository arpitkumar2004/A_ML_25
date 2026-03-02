# src/features/image_embeddings.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Iterable, List
import os
import joblib
from ..utils.io import IO
from ..utils.logging_utils import LoggerFactory
from ..utils.timer import timer

logger = LoggerFactory.get("image_embeddings")

# Try importing CLIP from openai/clip or transformers; implement flexible loader
try:
    import torch
    from PIL import Image
    from torchvision import transforms
    # try CLIP via sentence-transformers or openai; fall back to huggingface transformers
    try:
        import clip  # openai/clip (if installed)
        CLIP_AVAILABLE = "openai_clip"
    except Exception:
        try:
            from transformers import CLIPProcessor, CLIPModel
            CLIP_AVAILABLE = "hf_clip"
        except Exception:
            CLIP_AVAILABLE = None
except Exception:
    CLIP_AVAILABLE = None
    torch = None

class ImageEmbedder:
    """
    Extract image embeddings using CLIP (preferred) or fallback.
    Supports image paths or URLs. Caches results to disk.
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 cache_path: Optional[str] = "data/processed/image_embeddings.joblib",
                 device: Optional[str] = None,
                 batch_size: int = 32):
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self._model = None
        self._processor = None
        self._init_model_if_possible()

    def _init_model_if_possible(self):
        if CLIP_AVAILABLE == "openai_clip":
            try:
                import clip
                self._model, _ = clip.load("ViT-B/32", device=self.device)
                self._model.eval()
                logger.info("Loaded openai clip ViT-B/32")
            except Exception as e:
                logger.warning(f"openai clip load failed: {e}")
                self._model = None
        elif CLIP_AVAILABLE == "hf_clip":
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model.eval()
                logger.info(f"Loaded HF CLIP model: {self.model_name}")
            except Exception as e:
                logger.warning(f"HF CLIP load failed: {e}")
                self._model = None
        else:
            logger.warning("No CLIP backend available; image embeddings will be zeros.")
            self._model = None

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached image embeddings from {self.cache_path}")
            return joblib.load(self.cache_path)
        return None

    def _save_cache(self, obj):
        if not self.cache_path:
            return
        IO.save_pickle(obj, self.cache_path)
        logger.info(f"Saved image embeddings to {self.cache_path}")

    def _preprocess_pil(self, pil_images: List["PIL.Image.Image"]):
        # basic preprocessing for CLIP-sized inputs
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        batch = torch.stack([preprocess(img).to(self.device) for img in pil_images])
        return batch

    def _load_image(self, path_or_url: str):
        # For now treat as local path; if URL detected, user should pre-download or extend
        from PIL import Image
        try:
            if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
                # lazy: try to download once (requests) - may fail if requests missing
                import requests, io
                r = requests.get(path_or_url, timeout=10)
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                img = Image.open(path_or_url).convert("RGB")
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {path_or_url}: {e}")
            # return a blank image
            from PIL import Image as PILImage
            return PILImage.new("RGB", (224, 224), color=(255,255,255))

    def embed(self, image_paths: Iterable[str], use_cache: bool = True):
        """
        Compute embeddings for each image path in image_paths (list-like).
        Returns numpy array: (n_images, emb_dim)
        """
        cached = self._load_cache() if use_cache else None
        if cached is not None:
            return cached

        img_paths = list(image_paths)
        n = len(img_paths)
        if self._model is None:
            # Return zeros matrix with fallback dimension 512
            logger.warning("No image model available; returning zeros embeddings.")
            emb = np.zeros((n, 512), dtype=np.float32)
            self._save_cache(emb)
            return emb

        embeddings = []
        with timer("image_embedding", logger=logger):
            for i in range(0, n, self.batch_size):
                batch_paths = img_paths[i:i+self.batch_size]
                pil_images = [self._load_image(p) for p in batch_paths]
                if CLIP_AVAILABLE == "openai_clip":
                    import clip
                    # preprocess with clip
                    preprocess = clip._transform
                    batch_input = torch.stack([preprocess(img).to(self.device) for img in pil_images])
                    with torch.no_grad():
                        emb_batch = self._model.encode_image(batch_input)
                        emb_batch = emb_batch.cpu().numpy()
                else:
                    # HF CLIP processor
                    inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self._model.get_image_features(**inputs)
                        emb_batch = outputs.cpu().numpy()
                # normalize
                norms = np.linalg.norm(emb_batch, axis=1, keepdims=True) + 1e-12
                emb_batch = emb_batch / norms
                embeddings.append(emb_batch)
        emb = np.vstack(embeddings)
        self._save_cache(emb)
        return emb

# Seperate line

# ==========================================
# Load model on GPU (if available)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()



# ==========================================
# Create a Dataset class for URLs
# ==========================================
class ImageURLDataset(Dataset):
    def __init__(self, df, url_col):
        self.urls = df[url_col].tolist()

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            # Fallback: return a blank image if download fails
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        return image

# ==========================================
# Collate function for batch processing
# ==========================================
def collate_fn(batch):
    return clip_processor(images=batch, return_tensors="pt", padding=True)

# ==========================================
# Function to generate embeddings
# ==========================================
def generate_image_embeddings(df, url_col="image_link", batch_size=32):
    dataset = ImageURLDataset(df, url_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔹 Generating image embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = clip_model.get_image_features(**batch)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # Normalize
            all_embeddings.append(emb.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_embeddings



# # ==========================================
# # Example usage
# # ==========================================
# # Assuming df is your dataframe with 'image_link' column
# df_processed = pd.read_csv("/content/test.csv")

# image_embeddings = generate_image_embeddings(df_processed, url_col="image_link", batch_size=64)

# # Save embeddings as numpy or attach to dataframe
# df_processed["image_embedding"] = list(image_embeddings)

# # Optional: save separately
# np.save("clip_image_embeddings.npy", image_embeddings)
# df_processed.to_parquet("df_with_clip_image_embeddings.parquet")

# print("Image embeddings generated and saved successfully!")
# print(f"Shape: {image_embeddings.shape}")
