import numpy as np
import pandas as pd
from scipy import sparse

from src.data.parse_features import Parser
from src.features.feature_selector import FeatureSelector


def test_parser_adds_normalized_quantity_features():
    df = pd.DataFrame(
        {
            "Description": [
                "Protein powder 2 lb + 16 oz value pack",
                "Shampoo 500 ml pack of 2",
                "No quantity mentioned",
            ]
        }
    )

    out = Parser.add_parsed_features(df, text_col="Description")

    assert "parsed_total_weight_g" in out.columns
    assert "parsed_total_volume_ml" in out.columns
    assert "parsed_total_count_units" in out.columns
    assert "parsed_quantity_mentions" in out.columns

    # Row 0: 2 lb + 16 oz should be substantial positive grams.
    assert out.loc[0, "parsed_total_weight_g"] > 1000
    # Row 1: explicit volume and count-like token.
    assert out.loc[1, "parsed_total_volume_ml"] >= 500
    assert out.loc[1, "parsed_total_count_units"] >= 2
    # Row 2: no mentions.
    assert out.loc[2, "parsed_has_quantity"] == 0.0


def test_feature_selector_fit_transform_and_inference_transform(tmp_path):
    rng = np.random.RandomState(42)
    X = rng.randn(50, 20)
    y = X[:, 0] * 2.0 + X[:, 1] * -0.5 + rng.randn(50) * 0.01

    selector_path = str(tmp_path / "selector.joblib")
    selector = FeatureSelector(method="f_regression", k=5, min_features=3, save_path=selector_path)
    X_sel, meta = selector.fit_transform(X, y)

    assert X_sel.shape[1] == 5
    assert meta["selected_count"] == 5

    # New instance simulates inference-time load.
    selector_infer = FeatureSelector(method="f_regression", k=5, min_features=3, save_path=selector_path)
    X_sparse = sparse.csr_matrix(X)
    X_sel_sparse = selector_infer.transform(X_sparse)
    assert X_sel_sparse.shape[1] == 5
