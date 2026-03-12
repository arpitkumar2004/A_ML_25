from src.data.dataset_loader import load_train_df
import pandas as pd
def test_loader(tmp_path):
    df = pd.DataFrame({"unique_identifier":[1], "Description":["a"], "Price":[10.0]})
    p = tmp_path/"train.csv"
    df.to_csv(p, index=False)
    loaded = load_train_df(str(p))
    assert "catalog_content" in loaded.columns
