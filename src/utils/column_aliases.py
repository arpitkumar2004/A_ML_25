from typing import Dict, Iterable, List, Tuple

import pandas as pd


TRAIN_SCHEMA_DEFAULTS: Dict[str, str] = {
    "id_col": "sample_id",
    "text_col": "catalog_content",
    "image_col": "image_link",
    "target_col": "price",
}


_ALIAS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "sample_id": ("sample_id", "unique_identifier"),
    "catalog_content": ("catalog_content", "Description"),
    "image_link": ("image_link", "image_path"),
    "price": ("price", "Price"),
}


def _group_for(column_name: str) -> Tuple[str, ...]:
    for group in _ALIAS_GROUPS.values():
        if column_name in group:
            return group
    return (column_name,)


def _candidate_alias_names(column_name: str) -> Tuple[str, ...]:
    candidates: List[str] = [column_name]

    for group in _ALIAS_GROUPS.values():
        for alias in group:
            if column_name == alias or column_name.startswith(f"{alias}_"):
                suffix = column_name[len(alias):]
                for candidate in group:
                    derived_name = f"{candidate}{suffix}"
                    if derived_name not in candidates:
                        candidates.append(derived_name)

    return tuple(candidates)


def resolve_column_name(columns: Iterable[str], requested: str) -> str:
    available = set(columns)
    for candidate in _candidate_alias_names(requested):
        if candidate in available:
            return candidate
    for candidate in _group_for(requested):
        if candidate in available:
            return candidate
    return requested


def normalize_to_train_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Rename known aliases to raw-train defaults when needed.

    Example: Description -> catalog_content, Price -> price.
    """
    rename_map: Dict[str, str] = {}
    for target_name, aliases in _ALIAS_GROUPS.items():
        if target_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def missing_required_columns(columns: Iterable[str], required_columns: List[str]) -> List[str]:
    missing: List[str] = []
    available = set(columns)
    for required in required_columns:
        resolved = resolve_column_name(available, required)
        if resolved not in available:
            missing.append(required)
    return missing
