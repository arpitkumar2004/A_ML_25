# Clip extractor model
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
# Load model on GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()

class ClipExtractor:
    def __init__(self):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device

    def _fetch_image(self, url):
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            # Fallback: return a blank image if download fails
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        return image

    def _get_image_embeddings(self, images):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for img in images:
                inputs = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.get_image_features(**inputs)
                emb = outputs.cpu().numpy()
                embeddings.append(emb)
        return np.vstack(embeddings)

    def extract(self, df, url_col="image_link", batch_size=32):
        dataset = [self._fetch_image(url) for url in df[url_col].tolist()]
        all_embeddings = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating image embeddings"):
            batch_images = dataset[i:i+batch_size]
            emb = self._get_image_embeddings(batch_images)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        return self    
    