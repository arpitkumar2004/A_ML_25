"""Feature parsing helpers (counts, weights)."""
# src/data/parse_features.py
from typing import Optional, Dict, List, Tuple
import re
import pandas as pd
import numpy as np


class Parser:
    """
    Extract structured pieces from description text: value and unit parsing,
    counts of items, ounces, weights, etc.
    """

    UNIT_RULES = {
        # weight -> grams
        "mg": ("weight_g", 0.001),
        "milligram": ("weight_g", 0.001),
        "milligrams": ("weight_g", 0.001),
        "g": ("weight_g", 1.0),
        "gram": ("weight_g", 1.0),
        "grams": ("weight_g", 1.0),
        "kg": ("weight_g", 1000.0),
        "kilogram": ("weight_g", 1000.0),
        "kilograms": ("weight_g", 1000.0),
        "oz": ("weight_g", 28.349523125),
        "ounce": ("weight_g", 28.349523125),
        "ounces": ("weight_g", 28.349523125),
        "lb": ("weight_g", 453.59237),
        "lbs": ("weight_g", 453.59237),
        "pound": ("weight_g", 453.59237),
        "pounds": ("weight_g", 453.59237),
        # volume -> milliliters
        "ml": ("volume_ml", 1.0),
        "milliliter": ("volume_ml", 1.0),
        "milliliters": ("volume_ml", 1.0),
        "l": ("volume_ml", 1000.0),
        "liter": ("volume_ml", 1000.0),
        "liters": ("volume_ml", 1000.0),
        "fl oz": ("volume_ml", 29.5735295625),
        "floz": ("volume_ml", 29.5735295625),
        # count-like
        "ct": ("count_units", 1.0),
        "count": ("count_units", 1.0),
        "pcs": ("count_units", 1.0),
        "pc": ("count_units", 1.0),
        "piece": ("count_units", 1.0),
        "pieces": ("count_units", 1.0),
        "pack": ("count_units", 1.0),
        "packs": ("count_units", 1.0),
        "serving": ("count_units", 1.0),
        "servings": ("count_units", 1.0),
        "bar": ("count_units", 1.0),
        "bars": ("count_units", 1.0),
    }

    QUANTITY_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(fl\s*oz|floz|mg|g|kg|oz|ounce|ounces|lb|lbs|pound|pounds|ml|l|liter|liters|milliliter|milliliters|ct|count|pcs|pc|piece|pieces|pack|packs|serving|servings|bar|bars)\b",
        flags=re.I,
    )
    UNIT_FIRST_PATTERN = re.compile(
        r"(pack|packs|ct|count|pcs|pc|piece|pieces|serving|servings|bar|bars)\s*(?:of\s*)?(\d+(?:\.\d+)?)\b",
        flags=re.I,
    )

    @staticmethod
    def _extract_quantities(text: Optional[str]) -> List[Tuple[float, str]]:
        if not text:
            return []
        matches = Parser.QUANTITY_PATTERN.findall(text)
        quantities = []
        for value, unit in matches:
            try:
                quantities.append((float(value), unit.lower().replace(" ", "")))
            except Exception:
                continue
        unit_first_matches = Parser.UNIT_FIRST_PATTERN.findall(text)
        for unit, value in unit_first_matches:
            try:
                quantities.append((float(value), unit.lower().replace(" ", "")))
            except Exception:
                continue
        return quantities

    @staticmethod
    def _normalized_quantity_stats(text: Optional[str]) -> Dict[str, float]:
        stats = {
            "total_weight_g": 0.0,
            "total_volume_ml": 0.0,
            "total_count_units": 0.0,
            "quantity_mentions": 0.0,
            "has_quantity": 0.0,
        }
        quantities = Parser._extract_quantities(text)
        if not quantities:
            return stats

        stats["quantity_mentions"] = float(len(quantities))
        stats["has_quantity"] = 1.0
        for value, unit in quantities:
            mapped = Parser.UNIT_RULES.get(unit)
            if mapped is None:
                continue
            bucket, scale = mapped
            stats[f"total_{bucket}"] = stats.get(f"total_{bucket}", 0.0) + (value * scale)

        return stats

    @staticmethod
    def parse_ounces(text: Optional[str]) -> float:
        if not text:
            return 0.0
        # match patterns like "7 oz", "7.0 oz", "7oz"
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*oz\b", text, flags=re.I)
        try:
            return float(sum(map(float, matches))) if matches else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def parse_value_unit(text: Optional[str]) -> (float, str):
        """
        Attempt to extract first occurring value + unit pair, e.g. '7 oz', '2 pack', '500 g'.
        Returns (value, unit) or (0.0, '')
        """
        if not text:
            return 0.0, ""
        # regex to capture number + unit (word characters or % or g/ml/oz)
        match = re.search(r"(\d+(?:\.\d+)?)[\s\-]*(mg|g|kg|oz|ml|l|pack|count|ct|bars|pcs|pieces|servings|ounce|pound|lb|ozs)?\b", text, flags=re.I)
        if match:
            val = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            return val, unit.lower()
        return 0.0, ""

    @staticmethod
    def add_parsed_features(df: pd.DataFrame, text_col: str = "catalog_content") -> pd.DataFrame:
        df = df.copy()
        df[f"{text_col}_clean_len"] = df[text_col].fillna("").astype(str).apply(len)
        df["parsed_ounces"] = df[text_col].fillna("").astype(str).apply(Parser.parse_ounces)
        vals_units = df[text_col].fillna("").astype(str).apply(Parser.parse_value_unit)
        df["parsed_value"] = vals_units.apply(lambda x: x[0]).astype(float)
        df["parsed_unit"] = vals_units.apply(lambda x: x[1]).astype(str)
        quantity_stats = df[text_col].fillna("").astype(str).apply(Parser._normalized_quantity_stats).apply(pd.Series)
        quantity_stats = quantity_stats.rename(
            columns={
                "total_weight_g": "parsed_total_weight_g",
                "total_volume_ml": "parsed_total_volume_ml",
                "total_count_units": "parsed_total_count_units",
                "quantity_mentions": "parsed_quantity_mentions",
                "has_quantity": "parsed_has_quantity",
            }
        )
        df = pd.concat([df, quantity_stats], axis=1)

        # derived: value normalized (will convert units later)
        df["parsed_value_log1p"] = np.log1p(df["parsed_value"].fillna(0.0).astype(float))
        df["parsed_weight_log1p"] = np.log1p(df["parsed_total_weight_g"].fillna(0.0).astype(float))
        df["parsed_volume_log1p"] = np.log1p(df["parsed_total_volume_ml"].fillna(0.0).astype(float))
        df["parsed_count_log1p"] = np.log1p(df["parsed_total_count_units"].fillna(0.0).astype(float))
        return df


# Seperater line

def extract_catalog_fields(catalog_content):
    """
    Parses the catalog_content string and returns a dictionary of extracted fields.
    Combines the initial product description and structured sub-sections into a single
    'Product Description' field in the output dictionary.

    Args:
        catalog_content (str): The string containing catalog information.

    Returns:
        dict: A dictionary with extracted field names as keys and their values.
              Returns an empty dictionary if input is NaN.
    """
    if pd.isna(catalog_content):
        return {}

    extracted_data = {
        'Item Name': None,
        'Bullet Points': None,
        'Product Description': None,
        'Value': None,
        'Unit': None
    }
    initial_description = ""
    description_subsections = {}

    # Extract Item Name
    item_name_match = re.search(r'Item Name: (.*?)\n', catalog_content)
    if item_name_match:
        extracted_data['Item Name'] = item_name_match.group(1).strip()
        catalog_content = catalog_content[item_name_match.end():] # Remove processed part

    # Extract Bullet Points
    bullet_points = re.findall(r'Bullet Point \d+: (.*?)(?=\n(?:Bullet Point \d+:|Product Description:|Value:|Unit:|$))', catalog_content, re.DOTALL)
    if bullet_points:
        extracted_data['Bullet Points'] = [bp.strip() for bp in bullet_points]
        # Simple approach to remove processed parts for bullet points - assumes unique bullet point text
        for bp in bullet_points:
             catalog_content = catalog_content.replace(f'Bullet Point {bullet_points.index(bp) + 1}: {bp}', '', 1)


    # Extract Product Description, including initial part and sub-sections
    product_description_match = re.search(r'Product Description: (.*?)(?=\nValue:|\nUnit:|$)', catalog_content, re.DOTALL)
    if product_description_match:
        product_description_raw = product_description_match.group(1).strip()

        # Find the first <b> tag to separate initial description
        first_bold_match = re.search(r'<b>', product_description_raw)
        if first_bold_match:
            initial_description = product_description_raw[:first_bold_match.start()].strip()
            product_description_for_subsections = product_description_raw[first_bold_match.start():].strip()

            # Find all bolded titles and their content
            subsections_matches = re.findall(r'<b>(.*?)</b>(.*?(?=<b>|$))', product_description_for_subsections, re.DOTALL)

            if subsections_matches:
                for title, content in subsections_matches:
                    # Clean up title and content (remove <p> and remaining <b>/</b> if any)
                    cleaned_title = re.sub(r'<[pb/]*>', '', title).strip()
                    cleaned_content = re.sub(r'<[pb/]*>', '', content).strip()
                    description_subsections[cleaned_title] = cleaned_content
        else:
             # If no bold tags, the whole description is the initial description
             initial_description = re.sub(r'<[pb/]*>', '', product_description_raw).strip()

        # Combine initial description and subsections for the output
        combined_description = ""
        if initial_description:
            combined_description += initial_description + "\n\n"

        if description_subsections:
            for title, content in description_subsections.items():
                combined_description += f"**{title}:** {content}\n\n"

        extracted_data['Product Description'] = combined_description.strip()


        catalog_content = catalog_content[product_description_match.end():] # Remove processed part


    # Extract Value and Unit (assuming they are at the end)
    value_match = re.search(r'Value: (.*?)(?:\n|$)', catalog_content)
    if value_match:
        extracted_data['Value'] = value_match.group(1).strip()
        catalog_content = catalog_content.replace(value_match.group(0), '', 1)


    unit_match = re.search(r'Unit: (.*?)(?:\n|$)', catalog_content)
    if unit_match:
        extracted_data['Unit'] = unit_match.group(1).strip()
        catalog_content = catalog_content.replace(unit_match.group(0), '', 1)


    return extracted_data

# # Apply the function to the 'catalog_content' column and create a new DataFrame from the results
# parsed_data = df['catalog_content'].apply(extract_catalog_fields).apply(pd.Series)

# # Concatenate the new columns with the original DataFrame (optional, depending on desired output)
# # Merge the extracted data with the original DataFrame based on the index
# df_processed = pd.concat([df[['sample_id']], parsed_data], axis=1)
# df_processed = pd.concat([df_processed, df[['image_link', 'price']]], axis=1)

# # For now, let's display the first few rows of the new combined DataFrame
# display(df_processed.head())

class CatalogProcessor:
    """
    A class to process a catalog DataFrame by parsing text content,
    restructuring columns, and cleaning the data.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the processor with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to be processed.
                                      A copy is created to avoid side effects.
        """
        self.df = dataframe.copy()

    def _parse_catalog_entry(self, content: str) -> dict:
        """
        Helper method to extract structured fields from a single catalog text entry.
        This is an internal method and not intended to be called directly.
        """
        if not isinstance(content, str):
            return {
                "item_name": None,
                "bullet_points": [],
                "product_description": None,
                "value": None,
                "unit": None,
            }

        item_name = re.search(r"Item Name:\s*(.+)", content)
        value = re.search(r"Value:\s*([\d\.]+)", content)
        unit = re.search(r"Unit:\s*([A-Za-z]+)", content)
        bullets = re.findall(r"Bullet Point\s*\d+:\s*(.+)", content)
        product_description = re.search(r"Product Description:\s*(.+)", content)

        return {
            "item_name": item_name.group(1).strip() if item_name else None,
            "bullet_points": bullets,
            "product_description": (
                product_description.group(1).strip() if product_description else None
            ),
            "value": float(value.group(1)) if value else None,
            "unit": unit.group(1).strip() if unit else None,
        }

    def parse_catalog_column(self, column_name: str = "catalog_content"):
        """
        Parses the catalog text column and expands its content into new columns.
        """
        parsed_series = self.df[column_name].apply(self._parse_catalog_entry)
        parsed_df = parsed_series.apply(pd.Series)
        self.df = pd.concat([self.df, parsed_df], axis=1)
        return self  # Return self to allow method chaining

    def combine_bullet_points(
        self,
        column_name: str = "bullet_points",
        new_column_name: str = "bullet_points_str",
    ):
        """
        Combines the list of bullet points into a single semicolon-separated string.
        """
        if column_name in self.df.columns:
            self.df[new_column_name] = self.df[column_name].apply(
                lambda lst: "; ".join(lst) if isinstance(lst, list) and lst else ""
            )
        return self

    def drop_unnecessary_columns(self, columns_to_drop: list = None):
        """
        Drops specified columns from the DataFrame.
        """
        if columns_to_drop is None:
            # Default columns to drop if none are provided
            columns_to_drop = ["catalog_content", "image_link", "bullet_points"]

        cols_to_drop_existing = [
            col for col in columns_to_drop if col in self.df.columns
        ]
        self.df = self.df.drop(columns=cols_to_drop_existing, errors="ignore")
        return self

    def move_target_column_to_end(self, target_column: str = "price"):
        """
        Moves a target column (e.g., the prediction target) to the end of the DataFrame.
        """
        if target_column not in self.df.columns:
            print(f"Warning: Column '{target_column}' not found. Skipping this step.")
            return self

        cols = [col for col in self.df.columns if col != target_column] + [
            target_column
        ]
        self.df = self.df[cols]
        return self

    def process(self) -> pd.DataFrame:
        """
        Executes the full data processing pipeline in a single call.

        Returns:
            pd.DataFrame: The fully processed and cleaned DataFrame.
        """
        self.parse_catalog_column()
        self.combine_bullet_points()
        self.drop_unnecessary_columns()
        self.move_target_column_to_end()
        return self.df


# --- Example Usage ---

# # 1. Create a sample DataFrame
# data = {
#     "product_id": [101, 102, 103],
#     "price": [19.99, 25.50, 9.75],
#     "image_link": ["link1.jpg", "link2.jpg", "link3.jpg"],
#     "catalog_content": [
#         "Item Name: Wireless Mouse\nBullet Point 1: Ergonomic Design\nBullet Point 2: 2.4GHz Wireless\nProduct Description: A comfortable and reliable wireless mouse.\nValue: 85\nUnit: grams",
#         "Item Name: Mechanical Keyboard\nBullet Point 1: RGB Backlight\nProduct Description: A gaming keyboard with tactile switches.\nValue: 1.2\nUnit: kg",
#         "Invalid data format",  # Example of a row that can't be parsed
#     ],
# }
# sample_df = pd.DataFrame(data)

# # 2. Instantiate the processor with the DataFrame
# processor = CatalogProcessor(sample_df)

# # 3. Run the entire processing pipeline
# processed_df = processor.process()

# # 4. Display the result
# print(processed_df)


