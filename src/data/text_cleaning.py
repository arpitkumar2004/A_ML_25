import re
import pandas as pd


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

# 1. Create a sample DataFrame
data = {
    "product_id": [101, 102, 103],
    "price": [19.99, 25.50, 9.75],
    "image_link": ["link1.jpg", "link2.jpg", "link3.jpg"],
    "catalog_content": [
        "Item Name: Wireless Mouse\nBullet Point 1: Ergonomic Design\nBullet Point 2: 2.4GHz Wireless\nProduct Description: A comfortable and reliable wireless mouse.\nValue: 85\nUnit: grams",
        "Item Name: Mechanical Keyboard\nBullet Point 1: RGB Backlight\nProduct Description: A gaming keyboard with tactile switches.\nValue: 1.2\nUnit: kg",
        "Invalid data format",  # Example of a row that can't be parsed
    ],
}
sample_df = pd.DataFrame(data)

# 2. Instantiate the processor with the DataFrame
processor = CatalogProcessor(sample_df)

# 3. Run the entire processing pipeline
processed_df = processor.process()

# 4. Display the result
print(processed_df)
