import importlib


def test_root_ui_exposes_sample_pipeline_data():
    serving_module = importlib.import_module("src.serving.app")

    body = serving_module.root()

    assert "NEURALIS" in body
    assert "Calculate Valuation" in body
    assert "Product Information" in body
    assert "Valuation Driver Breakdown" in body
    assert "/frontend/dashboard.css" in body
    assert "/frontend/dashboard.js" in body

