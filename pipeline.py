"""
Multi-Modal Image Classification Pipeline
Integrates OpenAI CLIP to validate, generate, and classify e-commerce imagery.
"""

import json
import os
import requests
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION LOADER ---
# Checks for a local config file first
CONFIG_PATH = "config.json"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Please create a {CONFIG_PATH} file. See config.example.json.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- SETUP DEVICE & MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading CLIP model on {device.upper()}...")

model = CLIPModel.from_pretrained(config["vision_model"]["name"]).to(device)
processor = CLIPProcessor.from_pretrained(config["vision_model"]["name"])

# Pre-compute label embeddings
labels = config["vision_model"]["labels"]
inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    encoded_text = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# --- CORE FUNCTIONS ---

def generate_variants(base_url, suffixes):
    """Generates variant URLs by stripping existing suffixes and appending new ones."""
    if not isinstance(base_url, str) or not base_url: return []
    
    # Simple logic to find the 'root' of the URL
    root_url = base_url.rsplit("_", 1)[0] if "_" in base_url else base_url.rsplit(".", 1)[0]
    
    variants = []
    for sfx in suffixes:
        new_url = root_url + sfx
        if new_url.endswith(".jp"): new_url += "g" # Fix .jp extension
        if new_url != base_url: variants.append(new_url)
            
    return variants

def download_image(url):
    """Downloads image with a short timeout. Returns PIL Image or None."""
    try:
        resp = requests.get(url, timeout=1, stream=True)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except:
        pass
    return None

def process_pipeline():
    # Load Data
    input_file = config["io"]["input_file"]
    print(f"📂 Reading: {input_file}")
    df = pd.read_csv(input_file)
    
    lifestyle_col = []
    product_col = []

    # Processing Loop
    print(f"⚡ Processing {len(df)} items...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        # 1. Generate
        src_link = row.get(config["csv_structure"]["input_url_column"], "")
        candidates = generate_variants(src_link, config["generation"]["suffixes"])
        
        # 2. Validate
        valid_imgs = []
        valid_urls = []
        for url in candidates:
            img = download_image(url)
            if img:
                valid_imgs.append(img)
                valid_urls.append(url)
        
        # 3. Classify
        l_list, p_list = [], []
        if valid_imgs:
            inputs = processor(images=valid_imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                probs = (model.get_image_features(**inputs) @ encoded_text.T).softmax(dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()
            
            for i, label_idx in enumerate(preds):
                if label_idx == 1: l_list.append(valid_urls[i])
                else: p_list.append(valid_urls[i])

        lifestyle_col.append(",".join(l_list))
        product_col.append(",".join(p_list))

    # Save
    df[config["csv_structure"]["output_lifestyle"]] = lifestyle_col
    df[config["csv_structure"]["output_product"]] = product_col
    
    output_file = config["io"]["output_file"]
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    process_pipeline()