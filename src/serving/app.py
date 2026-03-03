import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..inference.predict import PredictPipeline
from ..inference.postprocess import Postprocessor


def _load_env_file() -> None:
    """Best-effort .env loader for local serving runs.

    This avoids requiring python-dotenv and makes `uvicorn src.serving.app:app`
    pick up tokens/config from project-root `.env`.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)

    # Backward-compatible aliasing used by some HF clients/tools
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)


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

        text_method = os.getenv("TEXT_METHOD", "tfidf").lower()
        text_cfg = {
            "method": text_method,
            "cache_path": None,
            "tfidf_max_features": int(os.getenv("TFIDF_MAX_FEATURES", "1024")),
            "tfidf_ngram_range": (1, 2),
        }
        image_cfg = {
            "cache_path": None,
            "model_name": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
            "batch_size": int(os.getenv("IMAGE_BATCH_SIZE", "16")),
        }

        self.pipeline = PredictPipeline(
            text_cfg=text_cfg,
            image_cfg=image_cfg,
            numeric_cfg=None,
            feature_cache=None,
            dim_cache=os.getenv("DIM_CACHE", "data/processed/dimred.joblib"),
            models_dir=models_dir,
            oof_meta_path=os.getenv("OOF_META_PATH", "experiments/oof/model_names.joblib"),
            stacker_path=os.getenv("STACKER_PATH", "experiments/models/stacker.joblib"),
        )

        if not os.path.isdir(models_dir):
            self.ready = False
            self.ready_reason = f"models_dir_not_found:{models_dir}"
            return

        model_files = [
            f
            for f in os.listdir(models_dir)
            if (f.endswith(".joblib") or f.endswith(".pkl"))
            and (f.startswith("fold_") or re.search(r"_fold\d+", f) is not None)
        ]
        if not model_files:
            self.ready = False
            self.ready_reason = f"no_fold_model_artifacts_in:{models_dir}"
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
    _load_env_file()
    service.initialize()

@app.get("/", response_class=HTMLResponse)
def root():
        return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>A_ML_25 • Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
        .wrap { max-width: 820px; margin: 32px auto; padding: 0 16px; }
        .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 18px; margin-bottom: 14px; }
        h1 { margin: 0 0 10px; font-size: 24px; }
        p { color: #94a3b8; margin-top: 0; }
        label { display: block; margin: 10px 0 6px; font-size: 13px; color: #cbd5e1; }
        input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 8px; border: 1px solid #374151; background: #0b1220; color: #e2e8f0; }
        textarea { min-height: 110px; resize: vertical; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        button { margin-top: 12px; background: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 600; }
        button:hover { background: #1d4ed8; }
        .status { font-size: 13px; margin-top: 8px; color: #93c5fd; }
        .ok { color: #4ade80; }
        .err { color: #fca5a5; white-space: pre-wrap; }
        .result { font-size: 18px; font-weight: bold; margin-top: 8px; }
        code { background: #020617; padding: 2px 6px; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h1>Price Prediction UI</h1>
            <p>Quick browser-based tester for <code>/v1/predict</code>. Use this instead of raw JSON in Swagger when you want a user-like flow.</p>
            <div class="status" id="readyStatus">Checking service readiness...</div>
        </div>

        <div class="card">
            <label for="uid">Unique Identifier</label>
            <input id="uid" type="number" value="1" />

            <label for="desc">Product Description</label>
            <textarea id="desc">Organic green tea bags, 20 count, natural flavor.</textarea>

            <div class="row">
                <div>
                    <label for="img">Image Path or URL (optional)</label>
                    <input id="img" type="text" placeholder="https://... or local/path.jpg" />
                </div>
                <div>
                    <label for="round">Round to cents</label>
                    <input id="round" type="text" value="true" />
                </div>
            </div>

            <button id="predictBtn">Predict Price</button>
            <div class="status" id="runStatus"></div>
        </div>

        <div class="card">
            <div>Predicted Price</div>
            <div class="result" id="predResult">—</div>
            <div class="err" id="errResult"></div>
        </div>
    </div>

    <script>
        async function checkReady() {
            const el = document.getElementById('readyStatus');
            try {
                const res = await fetch('/readyz');
                if (res.ok) {
                    const data = await res.json();
                    el.className = 'status ok';
                    el.textContent = `Service ready: ${data.reason}`;
                } else {
                    const data = await res.json();
                    el.className = 'status err';
                    el.textContent = `Service not ready: ${JSON.stringify(data.detail)}`;
                }
            } catch (e) {
                el.className = 'status err';
                el.textContent = `Readiness check failed: ${e}`;
            }
        }

        async function predict() {
            const run = document.getElementById('runStatus');
            const out = document.getElementById('predResult');
            const err = document.getElementById('errResult');
            out.textContent = '—';
            err.textContent = '';
            run.className = 'status';
            run.textContent = 'Sending request...';

            const uid = Number(document.getElementById('uid').value || '1');
            const desc = document.getElementById('desc').value || '';
            const img = document.getElementById('img').value || '';
            const roundRaw = (document.getElementById('round').value || 'true').toLowerCase().trim();
            const round = roundRaw === 'true';

            const payload = {
                records: [{
                    unique_identifier: uid,
                    Description: desc,
                    image_path: img
                }],
                text_col: 'Description',
                image_col: 'image_path',
                id_col: 'unique_identifier',
                pred_col: 'predicted_price',
                min_value: 0.0,
                round: round
            };

            try {
                const res = await fetch('/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();
                if (!res.ok) {
                    run.className = 'status err';
                    run.textContent = `Request failed (${res.status})`;
                    err.textContent = JSON.stringify(data, null, 2);
                    return;
                }

                const pred = data?.predictions?.[0]?.predicted_price;
                out.textContent = (pred !== undefined && pred !== null) ? `${pred}` : JSON.stringify(data);
                run.className = 'status ok';
                run.textContent = 'Prediction successful';
            } catch (e) {
                run.className = 'status err';
                run.textContent = 'Request error';
                err.textContent = String(e);
            }
        }

        document.getElementById('predictBtn').addEventListener('click', predict);
        checkReady();
    </script>
</body>
</html>
"""

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
        preds = service.pipeline.predict(
            df,
            text_col=req.text_col,
            image_col=req.image_col,
            force_rebuild_features=True,
        )
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
