import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..inference.predict import PredictPipeline
from ..inference.postprocess import Postprocessor


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    text_col: str = "Description"
    image_col: str = "image_path"
    id_col: str = "unique_identifier"
    pred_col: str = "predicted_price"
    target_transform: Optional[str] = None
    min_value: float = 0.0
    round: bool = True


class ModelService:
    def __init__(self):
        self.ready: bool = False
        self.ready_reason: str = "initializing"
        self.pipeline: Optional[PredictPipeline] = None

    def initialize(self):
        models_dir = os.getenv("MODELS_DIR", "experiments/models")
        self.pipeline = PredictPipeline(
            text_cfg=None,
            image_cfg=None,
            numeric_cfg=None,
            feature_cache=os.getenv("FEATURE_CACHE", "data/processed/features.joblib"),
            dim_cache=os.getenv("DIM_CACHE", "data/processed/dimred.joblib"),
            models_dir=models_dir,
            oof_meta_path=os.getenv("OOF_META_PATH", "experiments/oof/model_names.joblib"),
            stacker_path=os.getenv("STACKER_PATH", "experiments/models/stacker.joblib"),
        )

        if not os.path.isdir(models_dir):
            self.ready = False
            self.ready_reason = f"models_dir_not_found:{models_dir}"
            return

        model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib") or f.endswith(".pkl")]
        if not model_files:
            self.ready = False
            self.ready_reason = f"no_model_artifacts_in:{models_dir}"
            return

        self.ready = True
        self.ready_reason = "ok"

    def warmup(self):
        if self.pipeline is None:
            raise RuntimeError("pipeline_not_initialized")
        self.pipeline._discover_base_models()
        self.pipeline._load_stacker()


service = ModelService()
app = FastAPI(title="A_ML_25 Inference Service", version="1.0.0")


@app.on_event("startup")
def on_startup():
    service.initialize()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if service.ready:
        return {"ready": True, "reason": service.ready_reason}
    raise HTTPException(status_code=503, detail={"ready": False, "reason": service.ready_reason})


@app.post("/v1/warmup")
def warmup():
    if not service.ready:
        raise HTTPException(status_code=503, detail={"ready": False, "reason": service.ready_reason})
    try:
        service.warmup()
        return {"warmed": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"warmup_failed:{exc}")


@app.post("/v1/predict")
def predict(req: PredictRequest):
    if not service.ready or service.pipeline is None:
        raise HTTPException(status_code=503, detail={"ready": False, "reason": service.ready_reason})

    try:
        df = pd.DataFrame(req.records)
        preds = service.pipeline.predict(df, text_col=req.text_col, image_col=req.image_col)
        if req.target_transform == "log1p":
            preds = Postprocessor.invert_log1p(preds)
        preds = Postprocessor.clip_min(preds, req.min_value)
        if req.round:
            preds = Postprocessor.round_to_cents(preds)

        if req.id_col in df.columns:
            out_df = Postprocessor.to_submission_df(df[req.id_col].values, preds, id_col=req.id_col, pred_col=req.pred_col)
            return {"predictions": out_df.to_dict(orient="records")}
        return {"predictions": preds.tolist()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction_failed:{exc}")
