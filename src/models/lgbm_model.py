"""Backward-compatibility shim for legacy pickled models.

Some historical artifacts were serialized with module path:
    src.models.lgbm_model

Current implementation lives in:
    src.models.lgb_model
"""

from .lgb_model import LGBModel


class LGBMModel(LGBModel):
    """Legacy class alias for old artifacts."""


__all__ = ["LGBModel", "LGBMModel"]
