import importlib


def test_root_ui_exposes_sample_pipeline_data():
    serving_module = importlib.import_module("src.serving.app")

    body = serving_module.root()

    assert "A_ML_25: Multimodal Pricing Engine" in body
    assert "Execute Serving Pipeline" in body
    assert "Execution Trace" in body
    assert "Ensemble Breakdown" in body
    assert "View Training Lineage" in body
    assert "/frontend/dashboard.css" in body
    assert "/frontend/dashboard.js" in body
