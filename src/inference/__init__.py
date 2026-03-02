# src/inference/__init__.py
"""Inference package.

Avoid importing heavy runtime dependencies at package import time.
"""

__all__ = ["PredictPipeline", "Postprocessor"]
