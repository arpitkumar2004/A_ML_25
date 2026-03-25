import pandas as pd

from src.utils.column_aliases import resolve_column_name


def test_resolve_column_name_supports_derived_alias_columns():
    df = pd.DataFrame(
        {
            "catalog_content_clean_len": [12],
            "sample_id": [1],
            "image_link": [""],
            "price": [10.0],
        }
    )

    assert resolve_column_name(df.columns, "Description_clean_len") == "catalog_content_clean_len"
    assert resolve_column_name(df.columns, "unique_identifier") == "sample_id"
    assert resolve_column_name(df.columns, "Price") == "price"
