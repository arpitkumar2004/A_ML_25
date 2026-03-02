def unit_emid(df):
  import re
  import numpy as np
  import pandas as pd

  # -------------------------------------------------------
  # 1️⃣ UNIT NORMALIZATION MAP
  # -------------------------------------------------------
  UNIT_MAP = {
      # --- Volume ---
      "fl oz": "floz",
      "fluid ounce": "floz",
      "fluid ounces": "floz",
      "fl.oz": "floz",
      "fl oz.": "floz",
      "fl ounce": "floz",
      "fluidounce": "floz",
      "ltr": "l",
      "liter": "l",
      "liters": "l",
      "litre": "l",
      "litres": "l",
      "milliliter": "ml",
      "millilitre": "ml",
      "mililitro": "ml",
      "ml": "ml",

      # --- Weight ---
      "oz": "oz",
      "ounce": "oz",
      "ounces": "oz",
      "gram": "g",
      "grams": "g",
      "gramm": "g",
      "gr": "g",
      "g": "g",
      "kg": "kg",
      "pound": "lb",
      "pounds": "lb",
      "lb": "lb",
      "lbs": "lb",

      # --- Count / pieces ---
      "count": "count",
      "ct": "count",
      "each": "count",
      "piece": "count",
      "pieces": "count",
      "unit": "count",
      "units": "count",
      "bottle": "count",
      "bottles": "count",
      "jar": "count",
      "can": "count",
      "cans": "count",
      "capsule": "count",
      "bag": "count",
      "bags": "count",
      "pouch": "count",
      "pack": "pack",
      "packs": "pack",
      "box": "pack",
      "boxes": "pack",
      "carton": "pack",
      "bucket": "pack",
      "k-cups": "pack",
      "per box": "pack",
      "per package": "pack",
      "per carton": "pack",
      "box/12": "pack",

      # --- Area / Length ---
      "sq ft": "sqft",
      "foot": "ft",
      "ft": "ft",
      "in": "inch",
  }

  # -------------------------------------------------------
  # 2️⃣ NORMALIZE UNIT FUNCTION
  # -------------------------------------------------------
  def normalize_unit(unit):
      """
      Cleans and normalizes a unit string to a consistent form.
      """
      if not isinstance(unit, str) or unit.strip() == "" or unit.lower() in {"none", "---", "-", "product_weight"}:
          return "unknown"

      u = unit.lower().strip()
      u = re.sub(r"[^a-z0-9 ]+", "", u)  # remove punctuation

      # Match known variants
      for key, val in UNIT_MAP.items():
          if key in u:
              return val

      # Numeric-only entries like "24"
      if re.fullmatch(r"\d+", u):
          return "count"

      return "unknown"

  # -------------------------------------------------------
  # 3️⃣ CONVERSION TO BASE UNIT
  # -------------------------------------------------------
  def convert_to_base_with_unit(value, unit):
      """
      Convert (value, unit) into a consistent base and return:
      (converted_value, base_unit)
      """
      try:
          value = float(value)
      except:
          return np.nan, "unknown"

      unit = str(unit).lower().strip()

      # --- Volume ---
      if unit == "floz":
          return value * 29.5735, "ml"
      elif unit == "ml":
          return value, "ml"
      elif unit == "l":
          return value * 1000, "ml"

      # --- Weight ---
      elif unit == "oz":
          return value * 28.3495, "g"
      elif unit == "lb":
          return value * 453.592, "g"
      elif unit == "g":
          return value, "g"

      # --- Length ---
      elif unit == "inch":
          return value, "inch"
      elif unit == "ft":
          return value * 12, "inch"

      # --- Area ---
      elif unit == "sqft":
          return value, "sqft"

      # --- Count, Pack, Unknown ---
      elif unit in ["count", "pack", "unknown"]:
          return value, unit

      return value, "unknown"


  # -------------------------------------------------------
  # 4️⃣ APPLY TO DATAFRAME
  # -------------------------------------------------------

  # Example: your dataframe columns
  # df.columns = ['sample_id', 'Item Name', 'Bullet Points', 'Product Description',
  #               'Value', 'Unit', 'image_link', 'price']

  # Normalize unit column
  df["unit_clean"] = df["Unit"].apply(normalize_unit)

  # Convert to base value and unit
  df[["value_converted", "base_unit"]] = df.apply(
      lambda x: pd.Series(convert_to_base_with_unit(x["Value"], x["unit_clean"])),
      axis=1
  )

  # Create measure text for embeddings
  df["measure_text"] = df["value_converted"].astype(str) + " " + df["base_unit"]

  # -------------------------------------------------------
  # 5️⃣ CLEAR GPU CACHE IF YOU’LL CREATE NEW EMBEDDINGS
  # -------------------------------------------------------
  import gc, torch
  gc.collect()
  torch.cuda.empty_cache()

  print(df[["Value", "Unit", "unit_clean", "value_converted", "base_unit", "measure_text"]].head(10))

  # import torch
  # from transformers import CLIPProcessor, CLIPModel

  # device = "cuda" if torch.cuda.is_available() else "cpu"
  # model_name = "openai/clip-vit-base-patch32"

  # model = CLIPModel.from_pretrained(model_name).to(device)
  # processor = CLIPProcessor.from_pretrained(model_name)

  # measure_texts = df["measure_text"].tolist()
  # batch_size = 32
  # embeddings = []

  # for i in range(0, len(measure_texts), batch_size):
  #     batch = measure_texts[i:i+batch_size]
  #     inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
  #     with torch.no_grad():
  #         text_emb = model.get_text_features(**inputs)
  #     embeddings.append(text_emb.cpu())

  # measure_embeddings = torch.cat(embeddings).numpy()
  # df["measure_embedding"] = list(measure_embeddings)
  # return df

