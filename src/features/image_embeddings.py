import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm import tqdm


# Load model on GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()



# ==========================================
# 3️⃣ Create a Dataset class for URLs
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
# 4️⃣ Collate function for batch processing
# ==========================================
def collate_fn(batch):
    return clip_processor(images=batch, return_tensors="pt", padding=True)

# ==========================================
# 5️⃣ Function to generate embeddings
# ==========================================
def generate_image_embeddings(df, url_col="image_link", batch_size=32):
    dataset = ImageURLDataset(df, url_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating image embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = clip_model.get_image_features(**batch)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # Normalize
            all_embeddings.append(emb.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_embeddings

