import os
import re
import time
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..inference.predict import PredictPipeline
from ..inference.postprocess import Postprocessor
from ..utils.logging_utils import LoggerFactory
from ..monitoring.data_quality import validate_batch_quality
from ..utils.io import IO
from ..utils.registry_loader import RegistryLoader
from ..utils.model_bundle import resolve_bundle_path, validate_bundle

logger = LoggerFactory.get("serving_app")

_SERVICE_METRICS = {
    "request_count": 0,
    "error_count": 0,
    "predict_count": 0,
    "latencies_ms": [],
    "fallback_image_rows": 0,
    "fallback_text_rows": 0,
    "predict_rows": 0,
    "dq_pass_count": 0,
    "dq_reject_count": 0,
    "dq_quarantine_count": 0,
}


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


def _load_frontend_html() -> str:
    frontend_path = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    if frontend_path.exists():
        return frontend_path.read_text(encoding="utf-8")

    logger.warning("frontend_index_missing path=%s", frontend_path)
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>A_ML_25 UI Missing</title></head>
<body style="font-family: Arial, sans-serif; padding: 24px;">
<h1>Frontend file missing</h1>
<p>The app expected <code>frontend/index.html</code> next to the project root.</p>
</body>
</html>"""


def _github_repo_url() -> str:
    return os.getenv("GITHUB_REPO_URL", "https://github.com/arpitkumar2004/A_ML_25").rstrip("/")


def _dagshub_repo_url() -> str:
    configured = os.getenv("DAGSHUB_REPO_URL", "").strip()
    if configured:
        return configured.rstrip("/")

    host = str(os.getenv("DAGSHUB_HOST", "https://dagshub.com")).strip().rstrip("/")
    owner = str(os.getenv("DAGSHUB_REPO_OWNER", "arpitkumar2004")).strip()
    repo = str(os.getenv("DAGSHUB_REPO_NAME", "A_ML_25")).strip()
    if owner and repo:
        return f"{host}/{owner}/{repo}"
    return host


def _mlflow_run_url(tracking_uri: str, run_id: Optional[str]) -> str:
    base = str(tracking_uri or "").strip().rstrip("/")
    if not base:
        return ""
    if not run_id:
        return base
    return (
        f"{base}/#/experiments?"
        f"searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&runId={run_id}"
    )


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    text_col: str = "catalog_content"
    image_col: str = "image_link"
    id_col: str = "sample_id"
    pred_col: str = "predicted_price"
    target_transform: Optional[str] = None
    min_value: float = 0.0
    round: bool = True


class ModelService:
    def __init__(self):
        self.ready: bool = False
        self.ready_reason: str = "initializing"
        self.pipeline: Optional[PredictPipeline] = None
        self.canary_pipeline: Optional[PredictPipeline] = None
        self.canary_enabled: bool = False
        self.canary_percent: float = 0.0
        self.compare_canary: bool = False
        self.registry_dir: str = "experiments/registry"
        self.primary_run_id: Optional[str] = None
        self.primary_bundle_path: Optional[str] = None
        self.canary_run_id: Optional[str] = None
        self.canary_bundle_path: Optional[str] = None

    @staticmethod
    def _build_pipeline(models_dir: Optional[str] = None, bundle_path: Optional[str] = None, registry_dir: str = "experiments/registry") -> PredictPipeline:
        if bundle_path:
            return PredictPipeline(
                text_cfg={"cache_path": None},
                image_cfg={
                    "cache_path": None,
                    "model_name": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
                    "batch_size": int(os.getenv("IMAGE_BATCH_SIZE", "16")),
                    "lazy_init": True,
                },
                bundle_path=bundle_path,
                registry_dir=registry_dir,
            )

        if not models_dir:
            raise ValueError("models_dir or bundle_path must be provided")
        base_dir = os.path.dirname(models_dir.rstrip("/\\"))
        text_method = os.getenv("TEXT_METHOD", "tfidf").lower()
        tfidf_vectorizer_path = os.getenv(
            "TFIDF_VECTORIZER_PATH",
            os.path.join(base_dir, "data", "tfidf_vectorizer.joblib"),
        )
        numeric_scaler_path = os.getenv(
            "NUMERIC_SCALER_PATH",
            os.path.join(base_dir, "data", "numeric_scaler.joblib"),
        )
        selector_path = os.getenv(
            "FEATURE_SELECTOR_PATH",
            os.path.join(base_dir, "data", "feature_selector.joblib"),
        )
        selector_enabled_env = os.getenv("SELECTOR_ENABLED", "auto").strip().lower()
        if selector_enabled_env == "auto":
            selector_enabled = os.path.exists(selector_path)
        else:
            selector_enabled = selector_enabled_env == "true"

        text_cfg = {
            "method": text_method,
            "cache_path": None,
            "vectorizer_path": tfidf_vectorizer_path,
            "tfidf_max_features": int(os.getenv("TFIDF_MAX_FEATURES", "1024")),
            "tfidf_ngram_range": (1, 2),
        }
        image_cfg = {
            "cache_path": None,
            "model_name": os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
            "batch_size": int(os.getenv("IMAGE_BATCH_SIZE", "16")),
            "lazy_init": True,
        }
        return PredictPipeline(
            text_cfg=text_cfg,
            image_cfg=image_cfg,
            numeric_cfg={"scaler_path": numeric_scaler_path},
            selector_cfg={"enabled": selector_enabled, "save_path": selector_path},
            feature_cache=None,
            dim_cache=os.getenv("DIM_CACHE", "data/processed/dimred.joblib"),
            models_dir=models_dir,
            oof_meta_path=os.getenv("OOF_META_PATH", "experiments/oof/model_names.joblib"),
            stacker_path=os.getenv("STACKER_PATH", "experiments/models/stacker.joblib"),
        )

    @staticmethod
    def _has_fold_models(models_dir: str) -> bool:
        if not os.path.isdir(models_dir):
            return False
        model_files = [
            f
            for f in os.listdir(models_dir)
            if (f.endswith(".joblib") or f.endswith(".pkl"))
            and (f.startswith("fold_") or re.search(r"_fold\d+", f) is not None)
        ]
        return bool(model_files)

    def initialize(self):
        registry_dir = os.getenv("REGISTRY_DIR", "experiments/registry")
        self.registry_dir = registry_dir
        bundle_path = os.getenv("MODEL_BUNDLE_PATH", "").strip()
        configured_run_id = os.getenv("MODEL_RUN_ID", "").strip()

        if not bundle_path:
            run_id = configured_run_id
            if not run_id:
                try:
                    run_id = RegistryLoader(registry_dir=registry_dir).get_active_production_run_id() or ""
                except Exception:
                    run_id = ""
            if run_id:
                try:
                    bundle_path = resolve_bundle_path(run_id=run_id, registry_dir=registry_dir) or ""
                except Exception as exc:
                    self.ready = False
                    self.ready_reason = f"bundle_resolution_failed:{exc}"
                    return

        if bundle_path:
            validation = validate_bundle(bundle_path)
            if not validation["valid"]:
                self.ready = False
                self.ready_reason = ";".join(validation["problems"])
                return
            self.pipeline = self._build_pipeline(bundle_path=bundle_path, registry_dir=registry_dir)
            self.primary_bundle_path = bundle_path
            self.primary_run_id = configured_run_id or None
        else:
            models_dir = os.getenv("MODELS_DIR", "experiments/models")
            base_dir = os.path.dirname(models_dir.rstrip("/\\"))
            text_method = os.getenv("TEXT_METHOD", "tfidf").lower()
            tfidf_vectorizer_path = os.getenv("TFIDF_VECTORIZER_PATH", os.path.join(base_dir, "data", "tfidf_vectorizer.joblib"))
            numeric_scaler_path = os.getenv("NUMERIC_SCALER_PATH", os.path.join(base_dir, "data", "numeric_scaler.joblib"))
            selector_path = os.getenv("FEATURE_SELECTOR_PATH", os.path.join(base_dir, "data", "feature_selector.joblib"))
            selector_enabled_env = os.getenv("SELECTOR_ENABLED", "auto").strip().lower()
            selector_enabled = (os.path.exists(selector_path) if selector_enabled_env == "auto" else selector_enabled_env == "true")

            if text_method == "tfidf" and not os.path.exists(tfidf_vectorizer_path):
                self.ready = False
                self.ready_reason = f"missing_tfidf_vectorizer:{tfidf_vectorizer_path}"
                return

            if not os.path.exists(numeric_scaler_path):
                self.ready = False
                self.ready_reason = f"missing_numeric_scaler:{numeric_scaler_path}"
                return

            if selector_enabled and not os.path.exists(selector_path):
                self.ready = False
                self.ready_reason = f"missing_feature_selector:{selector_path}"
                return

            if not self._has_fold_models(models_dir):
                self.ready = False
                self.ready_reason = f"no_fold_model_artifacts_in:{models_dir}"
                return

            self.pipeline = self._build_pipeline(models_dir=models_dir, registry_dir=registry_dir)
            self.primary_bundle_path = None
            self.primary_run_id = configured_run_id or None

        if self.primary_bundle_path and not self.primary_run_id:
            try:
                loader = RegistryLoader(registry_dir=registry_dir)
                for entry in loader.list_runs():
                    if entry.get("bundle_path") == self.primary_bundle_path and entry.get("run_id"):
                        self.primary_run_id = str(entry.get("run_id"))
                        break
            except Exception:
                self.primary_run_id = None

        canary_bundle = os.getenv("CANARY_BUNDLE_PATH", "").strip()
        canary_run_id = os.getenv("CANARY_RUN_ID", "").strip()
        if not canary_bundle and canary_run_id:
            try:
                canary_bundle = resolve_bundle_path(run_id=canary_run_id, registry_dir=registry_dir) or ""
            except Exception:
                canary_bundle = ""

        canary_dir = os.getenv("CANARY_MODELS_DIR", "").strip()
        self.canary_percent = float(os.getenv("CANARY_PERCENT", "0"))
        self.compare_canary = os.getenv("CANARY_COMPARE", "false").strip().lower() == "true"
        if canary_bundle:
            validation = validate_bundle(canary_bundle)
            self.canary_enabled = validation["valid"] and self.canary_percent > 0
            if self.canary_enabled:
                self.canary_pipeline = self._build_pipeline(bundle_path=canary_bundle, registry_dir=registry_dir)
                self.canary_bundle_path = canary_bundle
                self.canary_run_id = canary_run_id or None
                logger.info("Canary enabled via bundle: path=%s percent=%s", canary_bundle, self.canary_percent)
            else:
                self.canary_pipeline = None
                self.canary_bundle_path = None
                self.canary_run_id = None
        else:
            self.canary_enabled = bool(canary_dir) and self.canary_percent > 0 and self._has_fold_models(canary_dir)
            if self.canary_enabled:
                self.canary_pipeline = self._build_pipeline(models_dir=canary_dir, registry_dir=registry_dir)
                self.canary_bundle_path = None
                self.canary_run_id = canary_run_id or None
                logger.info("Canary enabled: dir=%s percent=%s", canary_dir, self.canary_percent)
            else:
                self.canary_pipeline = None
                self.canary_bundle_path = None
                self.canary_run_id = None

        if self.canary_bundle_path and not self.canary_run_id:
            try:
                loader = RegistryLoader(registry_dir=registry_dir)
                for entry in loader.list_runs():
                    if entry.get("bundle_path") == self.canary_bundle_path and entry.get("run_id"):
                        self.canary_run_id = str(entry.get("run_id"))
                        break
            except Exception:
                self.canary_run_id = None

        self.ready = True
        self.ready_reason = f"ok:bundle={bundle_path}" if bundle_path else "ok:legacy_models_dir"

    def choose_pipeline(self) -> (PredictPipeline, str):
        if self.pipeline is None:
            raise RuntimeError("pipeline_not_initialized")
        if self.canary_enabled and self.canary_pipeline is not None:
            if random.random() * 100.0 < self.canary_percent:
                return self.canary_pipeline, "canary"
        return self.pipeline, "primary"

    def warmup(self):
        if self.pipeline is None:
            raise RuntimeError("pipeline_not_initialized")
        self.pipeline._discover_base_models()
        self.pipeline._load_stacker()

    def deployment_state(self) -> Dict[str, Any]:
        registry_entry: Dict[str, Any] = {}
        try:
            if self.primary_run_id:
                registry_entry = RegistryLoader(registry_dir=self.registry_dir).get_run_by_id(self.primary_run_id) or {}
        except Exception:
            registry_entry = {}

        tracking = registry_entry.get("tracking", {}) if isinstance(registry_entry, dict) else {}
        mlflow_meta = tracking.get("mlflow", {}) if isinstance(tracking, dict) else {}
        mlflow_run_id = str(mlflow_meta.get("mlflow_run_id") or self.primary_run_id or "")
        tracking_uri = str(mlflow_meta.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI", "")).strip()
        environment_label = f"CANARY: {self.canary_percent:.0f}%" if self.canary_enabled and self.canary_percent > 0 else "PRODUCTION"

        return {
            "ready": self.ready,
            "reason": self.ready_reason,
            "api_status": "Healthy (Serving)" if self.ready else "Degraded",
            "environment": environment_label,
            "registry_dir": self.registry_dir,
            "run_id": self.primary_run_id,
            "bundle_path": self.primary_bundle_path,
            "tracking": {
                "mlflow_run_id": mlflow_run_id or None,
                "tracking_uri": tracking_uri or None,
                "experiment_name": mlflow_meta.get("experiment_name"),
            },
            "links": {
                "dagshub_repo": _dagshub_repo_url(),
                "github_repo": _github_repo_url(),
                "github_actions": f"{_github_repo_url()}/actions",
                "mlflow_run": _mlflow_run_url(tracking_uri=tracking_uri, run_id=mlflow_run_id),
            },
            "canary": {
                "enabled": self.canary_enabled,
                "percent": self.canary_percent,
                "compare": self.compare_canary,
                "run_id": self.canary_run_id,
                "bundle_path": self.canary_bundle_path,
            },
        }


service = ModelService()
app = FastAPI(title="A_ML_25 Inference Service", version="1.0.0")
app.mount(
    "/frontend",
    StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "frontend")),
    name="frontend",
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    _SERVICE_METRICS["request_count"] += 1

    api_key = os.getenv("API_KEY", "").strip()
    if api_key and request.url.path.startswith("/v1/"):
        provided = request.headers.get("x-api-key", "")
        if provided != api_key:
            _SERVICE_METRICS["error_count"] += 1
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Invalid API key", "request_id": request_id},
            )

    try:
        response = await call_next(request)
    except Exception as exc:
        _SERVICE_METRICS["error_count"] += 1
        logger.exception("request_failed request_id=%s path=%s err=%s", request_id, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": str(exc), "request_id": request_id},
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _SERVICE_METRICS["latencies_ms"].append(float(elapsed_ms))
    if len(_SERVICE_METRICS["latencies_ms"]) > 5000:
        _SERVICE_METRICS["latencies_ms"] = _SERVICE_METRICS["latencies_ms"][-5000:]

    response.headers["x-request-id"] = request_id
    return response


@app.on_event("startup")
def on_startup():
    _load_env_file()
    service.initialize()

@app.get("/", response_class=HTMLResponse)
def root():
    return _load_frontend_html()
    """
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
                    sample_id: uid,
                    catalog_content: desc,
                    image_link: img
                }],
                text_col: 'catalog_content',
                image_col: 'image_link',
                id_col: 'sample_id',
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
    state = service.deployment_state()
    return {"status": "ok" if state["ready"] else "degraded", "service": state}


@app.get("/metrics/json")
def metrics_json():
    lat = np.asarray(_SERVICE_METRICS["latencies_ms"], dtype=float)
    p50 = float(np.percentile(lat, 50)) if lat.size else 0.0
    p95 = float(np.percentile(lat, 95)) if lat.size else 0.0
    p99 = float(np.percentile(lat, 99)) if lat.size else 0.0
    requests_total = max(int(_SERVICE_METRICS["request_count"]), 1)
    predict_rows = max(int(_SERVICE_METRICS["predict_rows"]), 1)
    dq_total = int(_SERVICE_METRICS["dq_pass_count"] + _SERVICE_METRICS["dq_reject_count"] + _SERVICE_METRICS["dq_quarantine_count"])
    dq_total_safe = max(dq_total, 1)
    return {
        "request_count": int(_SERVICE_METRICS["request_count"]),
        "error_count": int(_SERVICE_METRICS["error_count"]),
        "predict_count": int(_SERVICE_METRICS["predict_count"]),
        "error_rate": float(_SERVICE_METRICS["error_count"]) / requests_total,
        "service": service.deployment_state(),
        "latency_ms": {"p50": p50, "p95": p95, "p99": p99},
        "canary": {
            "enabled": bool(service.canary_enabled),
            "percent": float(service.canary_percent),
            "compare": bool(service.compare_canary),
        },
        "fallback": {
            "image_rows": int(_SERVICE_METRICS["fallback_image_rows"]),
            "text_rows": int(_SERVICE_METRICS["fallback_text_rows"]),
            "predict_rows": int(_SERVICE_METRICS["predict_rows"]),
            "image_rate": float(_SERVICE_METRICS["fallback_image_rows"]) / predict_rows,
            "text_rate": float(_SERVICE_METRICS["fallback_text_rows"]) / predict_rows,
        },
        "data_quality": {
            "passed": int(_SERVICE_METRICS["dq_pass_count"]),
            "rejected": int(_SERVICE_METRICS["dq_reject_count"]),
            "quarantined": int(_SERVICE_METRICS["dq_quarantine_count"]),
            "pass_rate": float(_SERVICE_METRICS["dq_pass_count"]) / dq_total_safe,
        },
    }


@app.get("/metrics")
def metrics():
    lat = np.asarray(_SERVICE_METRICS["latencies_ms"], dtype=float)
    p50 = float(np.percentile(lat, 50)) if lat.size else 0.0
    p95 = float(np.percentile(lat, 95)) if lat.size else 0.0
    p99 = float(np.percentile(lat, 99)) if lat.size else 0.0

    model_run_id = service.primary_run_id or os.getenv("MODEL_RUN_ID", "unknown")
    model_version = os.getenv("MODEL_VERSION", "unknown")
    rows = max(int(_SERVICE_METRICS["predict_rows"]), 1)
    fallback_image_rate = float(_SERVICE_METRICS["fallback_image_rows"]) / rows
    fallback_text_rate = float(_SERVICE_METRICS["fallback_text_rows"]) / rows

    labels = f'run_id="{model_run_id}",model_version="{model_version}"'
    lines = [
        "# HELP request_count Total number of HTTP requests",
        "# TYPE request_count counter",
        f"request_count{{{labels}}} {_SERVICE_METRICS['request_count']}",
        "# HELP error_count Total number of errors",
        "# TYPE error_count counter",
        f"error_count{{{labels}}} {_SERVICE_METRICS['error_count']}",
        "# HELP predict_count Total number of prediction calls",
        "# TYPE predict_count counter",
        f"predict_count{{{labels}}} {_SERVICE_METRICS['predict_count']}",
        "# HELP latency_ms Latency percentiles in milliseconds",
        "# TYPE latency_ms gauge",
        f"latency_ms{{quantile=\"p50\",{labels}}} {p50}",
        f"latency_ms{{quantile=\"p95\",{labels}}} {p95}",
        f"latency_ms{{quantile=\"p99\",{labels}}} {p99}",
        "# HELP fallback_feature_rate Fraction of rows using fallback features",
        "# TYPE fallback_feature_rate gauge",
        f"fallback_feature_rate{{feature=\"image\",{labels}}} {fallback_image_rate}",
        f"fallback_feature_rate{{feature=\"text\",{labels}}} {fallback_text_rate}",
    ]
    body = "\n".join(lines) + "\n"
    return HTMLResponse(content=body, media_type="text/plain; version=0.0.4")


@app.get("/service/info")
def service_info():
    return service.deployment_state()


@app.get("/readyz")
def readyz():
    if service.ready:
        return service.deployment_state()
    raise HTTPException(status_code=503, detail=service.deployment_state())


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
        max_records = int(os.getenv("MAX_PREDICT_RECORDS", "200"))
        if len(req.records) > max_records:
            raise HTTPException(status_code=413, detail=f"too_many_records:{len(req.records)}>{max_records}")

        df = pd.DataFrame(req.records)

        n_rows = len(df)
        _SERVICE_METRICS["predict_rows"] += n_rows
        if req.image_col not in df.columns:
            _SERVICE_METRICS["fallback_image_rows"] += n_rows
        else:
            img_empty = df[req.image_col].isna() | (df[req.image_col].astype(str).str.strip() == "")
            _SERVICE_METRICS["fallback_image_rows"] += int(img_empty.sum())
        if req.text_col not in df.columns:
            _SERVICE_METRICS["fallback_text_rows"] += n_rows
        else:
            txt_empty = df[req.text_col].isna() | (df[req.text_col].astype(str).str.strip() == "")
            _SERVICE_METRICS["fallback_text_rows"] += int(txt_empty.sum())

        dq_rules = {
            "required_columns": [req.text_col, req.id_col],
            "max_null_rate": float(os.getenv("DQ_MAX_NULL_RATE", "0.5")),
            "reject_on": ["missing_columns", "out_of_bounds"],
            "numeric_bounds": {},
        }
        dq_result = validate_batch_quality(df, dq_rules)
        if dq_result.get("policy") == "reject":
            _SERVICE_METRICS["dq_reject_count"] += 1
            raise HTTPException(status_code=422, detail={"error": "data_quality_reject", "dq": dq_result})
        if dq_result.get("policy") == "quarantine":
            _SERVICE_METRICS["dq_quarantine_count"] += 1
            quarantine_path = os.getenv("SERVING_QUARANTINE_PATH", "experiments/quarantine/serving_quarantine.csv")
            IO.to_csv(df, quarantine_path, index=False)
            raise HTTPException(status_code=422, detail={"error": "data_quality_quarantine", "dq": dq_result, "path": quarantine_path})
        _SERVICE_METRICS["dq_pass_count"] += 1

        primary_pipeline, variant = service.choose_pipeline()
        preds, trace = primary_pipeline.predict(
            df,
            text_col=req.text_col,
            image_col=req.image_col,
            force_rebuild_features=True,
            return_diagnostics=True,
        )
        _SERVICE_METRICS["predict_count"] += 1

        divergence = None
        if service.compare_canary and service.canary_pipeline is not None and variant == "primary":
            try:
                canary_preds = service.canary_pipeline.predict(
                    df,
                    text_col=req.text_col,
                    image_col=req.image_col,
                    force_rebuild_features=True,
                )
                divergence = float(np.mean(np.abs(np.asarray(preds) - np.asarray(canary_preds))))
                logger.info("canary_compare divergence_mae=%s rows=%s", divergence, len(df))
            except Exception as canary_exc:
                logger.warning("canary_compare_failed err=%s", canary_exc)

        if req.target_transform == "log1p":
            preds = Postprocessor.invert_log1p(preds)
        preds = Postprocessor.clip_min(preds, req.min_value)
        if req.round:
            preds = Postprocessor.round_to_cents(preds)

        if req.id_col in df.columns:
            out_df = Postprocessor.to_submission_df(df[req.id_col].values, preds, id_col=req.id_col, pred_col=req.pred_col)
            return {
                "predictions": out_df.to_dict(orient="records"),
                "model_variant": variant,
                "canary_divergence_mae": divergence,
                "trace": trace,
            }
        return {
            "predictions": preds.tolist(),
            "model_variant": variant,
            "canary_divergence_mae": divergence,
            "trace": trace,
        }
    except Exception as exc:
        _SERVICE_METRICS["error_count"] += 1
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(exc)})
