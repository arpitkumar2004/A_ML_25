import numpy as np

from src.inference.postprocess import Postprocessor


def test_postprocess_clip_and_round():
    preds = np.array([-1.234, 3.4567])
    clipped = Postprocessor.clip_min(preds, min_value=0.0)
    rounded = Postprocessor.round_to_cents(clipped)
    assert rounded.tolist() == [0.0, 3.46]


def test_postprocess_submission_df_shape():
    ids = [10, 11, 12]
    preds = np.array([1.0, 2.0, 3.0])
    df = Postprocessor.to_submission_df(ids, preds, id_col="unique_identifier", pred_col="predicted_price")
    assert list(df.columns) == ["unique_identifier", "predicted_price"]
    assert len(df) == 3
