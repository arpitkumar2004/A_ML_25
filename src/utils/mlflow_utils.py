from typing import Any, Dict, Iterable, Optional
import os
import subprocess

from .logging_utils import LoggerFactory

logger = LoggerFactory.get("mlflow_utils")

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

try:
    import dagshub  # type: ignore
except Exception:
    dagshub = None

try:
    from mlflow.entities import ViewType  # type: ignore
except Exception:
    ViewType = None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _flatten_dict(data: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, parent=full_key, sep=sep))
        elif isinstance(value, (list, tuple)):
            out[full_key] = ",".join([str(v) for v in value])
        else:
            out[full_key] = value
    return out


class MLflowTracker:
    """Fail-safe MLflow wrapper for train/inference tracking."""

    def __init__(self, cfg: Dict[str, Any], stage: str, run_id: str):
        self.cfg = cfg or {}
        self.stage = stage
        self.local_run_id = run_id
        self.mlflow_run_id: Optional[str] = None
        self.enabled = self._is_enabled()

        mlflow_cfg = self.cfg.get("mlflow", {}) if isinstance(self.cfg.get("mlflow", {}), dict) else {}
        dagshub_cfg = mlflow_cfg.get("dagshub", {}) if isinstance(mlflow_cfg.get("dagshub", {}), dict) else {}

        self.tracking_uri = mlflow_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI") or ""
        self.experiment_name = mlflow_cfg.get("experiment_name") or os.getenv("MLFLOW_EXPERIMENT_NAME") or "A_ML_25"
        self.run_name = mlflow_cfg.get("run_name") or f"{self.stage}-{self.local_run_id}"

        self.dagshub_enabled = _to_bool(
            os.getenv("DAGSHUB_MLFLOW_ENABLED"),
            default=_to_bool(dagshub_cfg.get("enabled"), default=False),
        )
        self.dagshub_repo_owner = str(dagshub_cfg.get("repo_owner") or os.getenv("DAGSHUB_REPO_OWNER") or "").strip()
        self.dagshub_repo_name = str(dagshub_cfg.get("repo_name") or os.getenv("DAGSHUB_REPO_NAME") or "").strip()
        self.dagshub_host = str(dagshub_cfg.get("host") or os.getenv("DAGSHUB_HOST") or "https://dagshub.com").strip().rstrip("/")
        self.tracking_backend = "mlflow"
        model_registry_cfg = mlflow_cfg.get("model_registry", {}) if isinstance(mlflow_cfg.get("model_registry", {}), dict) else {}
        self.model_registry_enabled = _to_bool(model_registry_cfg.get("enabled"), default=True)
        self.model_registry_name = str(model_registry_cfg.get("name") or f"A_ML_25-{self.stage}-model").strip()
        self.dvc_rev = self._resolve_dvc_rev()

    def _resolve_dvc_rev(self) -> str:
        """Resolve the DVC data/code revision (Git SHA) for lineage tagging."""
        env_rev = str(os.getenv("DVC_REV") or "").strip()
        if env_rev:
            return env_rev

        if not os.path.exists(".dvc"):
            return ""

        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return ""

    def _is_enabled(self) -> bool:
        mlflow_cfg = self.cfg.get("mlflow", {}) if isinstance(self.cfg.get("mlflow", {}), dict) else {}
        cfg_enabled = mlflow_cfg.get("enabled")
        env_enabled = os.getenv("MLFLOW_ENABLED")
        enabled = _to_bool(cfg_enabled, default=True)
        if env_enabled is not None:
            enabled = _to_bool(env_enabled, default=enabled)
        if mlflow is None and enabled:
            logger.warning("MLflow is enabled but package import failed; tracking disabled.")
            return False
        return enabled

    def _init_dagshub_tracking(self) -> bool:
        if not self.dagshub_enabled:
            return False
        if dagshub is None:
            logger.warning("DagsHub tracking enabled but package import failed; using regular MLflow tracking.")
            return False
        if (not self.dagshub_repo_owner) or (not self.dagshub_repo_name):
            logger.warning("DagsHub tracking enabled but repo_owner/repo_name is missing; using regular MLflow tracking.")
            return False

        dagshub_token = os.getenv("DAGSHUB_TOKEN", "").strip()
        dagshub_username = os.getenv("DAGSHUB_USERNAME", "").strip()
        if (not dagshub_token) or (not dagshub_username):
            logger.warning("DagsHub tracking enabled but DAGSHUB_USERNAME/TOKEN is missing; using regular MLflow tracking.")
            return False

        # Directly configure the DagsHub MLflow endpoint using HTTP basic auth.
        # This avoids `dagshub.init()` which triggers an interactive browser OAuth flow in CI.
        dagshub_mlflow_uri = f"{self.dagshub_host}/{self.dagshub_repo_owner}/{self.dagshub_repo_name}.mlflow"
        try:
            assert mlflow is not None
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            mlflow.set_tracking_uri(dagshub_mlflow_uri)
            self.tracking_uri = dagshub_mlflow_uri
            self.tracking_backend = "dagshub"
            logger.info("DagsHub MLflow tracking configured: %s", dagshub_mlflow_uri)
            return True
        except Exception as exc:
            logger.warning("DagsHub MLflow setup failed: %s. Using regular MLflow tracking.", exc)
            return False

    def _set_or_restore_experiment(self) -> None:
        assert mlflow is not None
        try:
            mlflow.set_experiment(self.experiment_name)
            return
        except Exception as exc:
            message = str(exc)
            if "deleted experiment" not in message.lower():
                raise
            logger.warning("Experiment '%s' is deleted; attempting restore.", self.experiment_name)

        if ViewType is None:
            raise RuntimeError(
                f"Experiment '{self.experiment_name}' is deleted and MLflow ViewType is unavailable for restore."
            )

        client = mlflow.tracking.MlflowClient()
        deleted = client.search_experiments(view_type=ViewType.DELETED_ONLY)
        target = next((exp for exp in deleted if exp.name == self.experiment_name), None)
        if target is None:
            raise RuntimeError(
                f"Experiment '{self.experiment_name}' is deleted and could not be located for restore."
            )

        client.restore_experiment(target.experiment_id)
        mlflow.set_experiment(self.experiment_name)
        logger.info("Restored and activated deleted MLflow experiment '%s'.", self.experiment_name)

    def start(self) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            self._init_dagshub_tracking()
            # If DagsHub initialized successfully, keep its tracking URI.
            if self.tracking_backend != "dagshub" and self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            elif self.tracking_backend == "dagshub" and self.tracking_uri:
                logger.info("Ignoring MLFLOW_TRACKING_URI because DagsHub tracking is active.")

            resolved_uri = mlflow.get_tracking_uri()
            if isinstance(resolved_uri, str) and resolved_uri:
                self.tracking_uri = resolved_uri

            self._set_or_restore_experiment()
            run = mlflow.start_run(
                run_name=self.run_name,
                tags={
                    "pipeline_stage": self.stage,
                    "local_run_id": self.local_run_id,
                    "git_sha": os.getenv("GITHUB_SHA", ""),
                    "git_ref": os.getenv("GITHUB_REF", ""),
                    "dvc_rev": self.dvc_rev,
                    "tracking_backend": self.tracking_backend,
                },
            )
            self.mlflow_run_id = run.info.run_id
            logger.info("MLflow run started run_id=%s", self.mlflow_run_id)
        except Exception as exc:
            self.enabled = False
            logger.warning("Failed to start MLflow run: %s", exc)

    def log_config(self, cfg: Dict[str, Any], prefix: str = "config") -> None:
        flat = _flatten_dict(cfg)
        self.log_params({f"{prefix}.{k}": v for k, v in flat.items()})

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            cleaned = {str(k)[:250]: ("" if v is None else str(v))[:5000] for k, v in params.items()}
            mlflow.log_params(cleaned)
        except Exception as exc:
            logger.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            cleaned: Dict[str, float] = {}
            for k, v in metrics.items():
                try:
                    cleaned[str(k)] = float(v)
                except Exception:
                    continue
            if cleaned:
                mlflow.log_metrics(cleaned, step=step)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_artifact_if_exists(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.enabled or not path or (not os.path.exists(path)):
            return
        assert mlflow is not None
        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("MLflow log_artifact failed path=%s err=%s", path, exc)

    def log_artifacts_if_exists(self, paths: Iterable[str], artifact_path: Optional[str] = None) -> None:
        for p in paths:
            self.log_artifact_if_exists(p, artifact_path=artifact_path)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            mlflow.set_tags({str(k): "" if v is None else str(v) for k, v in tags.items()})
        except Exception as exc:
            logger.warning("MLflow set_tags failed: %s", exc)

    def end(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            mlflow.end_run(status=status)
        except Exception as exc:
            logger.warning("Failed to end MLflow run cleanly: %s", exc)

    def register_model_if_possible(self, model_obj: Any, model_key: str) -> Dict[str, Any]:
        """Log and register a trained model in MLflow Model Registry when available."""
        result = {
            "enabled": False,
            "registered": False,
            "registered_model_name": "",
            "version": None,
            "model_uri": "",
            "artifact_path": "",
            "error": "",
        }
        if (not self.enabled) or (not self.mlflow_run_id) or (not self.model_registry_enabled) or (model_obj is None):
            return result

        assert mlflow is not None
        try:
            base_model = getattr(model_obj, "model", model_obj)
            artifact_path = f"models/{str(model_key).lower()}"
            model_log_name = f"model_{str(model_key).lower()}"
            model_uri = ""

            # Prefer sklearn flavor when available; fallback to generic artifact log.
            logged = False
            try:
                from mlflow import sklearn as mlflow_sklearn  # type: ignore
                model_info = mlflow_sklearn.log_model(sk_model=base_model, name=model_log_name)
                model_uri = getattr(model_info, "model_uri", "") or f"runs:/{self.mlflow_run_id}/{model_log_name}"
                logged = True
            except Exception:
                try:
                    import joblib
                    tmp_path = os.path.join("experiments", "registry", f"{self.local_run_id}_{str(model_key).lower()}_model.joblib")
                    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                    joblib.dump(base_model, tmp_path)
                    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
                    logged = True
                except Exception as exc:
                    result["error"] = f"model_log_failed:{exc}"
                    return result

            if not logged:
                return result

            result.update(
                {
                    "enabled": True,
                    "artifact_path": artifact_path,
                    "model_uri": model_uri,
                    "registered_model_name": self.model_registry_name,
                }
            )

            if model_uri:
                try:
                    mv = mlflow.register_model(model_uri=model_uri, name=self.model_registry_name)
                    result["registered"] = True
                    result["version"] = getattr(mv, "version", None)
                except Exception as exc:
                    result["error"] = f"model_register_failed:{exc}"

            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result


def mlflow_link(run_id: Optional[str], tracking_uri: str, experiment_name: str) -> Dict[str, Any]:
    return {
        "enabled": bool(run_id),
        "mlflow_run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
    }
