import argparse
import os
import yaml
from datetime import datetime, timezone

from main import load_config
from src.utils.config_schema import flatten_train_config
from src.pipelines.train_pipeline import run_train_pipeline
from src.monitoring.drift_latency import load_latency_events, summarize_latency
from src.registry.model_registry import list_runs, promote_run
from src.utils.io import IO


def _load_policy(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("retrain_policy", cfg)


def _days_since(ts_iso: str) -> float:
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 86400.0


def _should_retrain(policy: dict):
    dashboard_path = policy.get("monitoring_dashboard_path", "experiments/monitoring/dashboard.json")
    latency_path = policy.get("latency_log_path", "experiments/monitoring/latency_events.jsonl")
    registry_dir = policy.get("registry_dir", "experiments/registry")
    triggers = policy.get("triggers", {})
    quality_gates = policy.get("quality_gates", {})

    drift_threshold = float(triggers.get("drift_psi_threshold", 0.3))
    latency_threshold = float(
        triggers.get(
            "latency_p95_seconds_threshold",
            quality_gates.get("max_p95_latency_seconds", 5.0),
        )
    )
    max_days = float(triggers.get("max_days_since_last_train", 7))
    max_critical = int(quality_gates.get("max_critical_drift_features", 0))

    reasons = []

    if os.path.exists(dashboard_path):
        report = IO.load_json(dashboard_path)
        top = report.get("drift_top", [])
        max_psi = max([float(r.get("psi", 0.0)) for r in top], default=0.0)
        if max_psi >= drift_threshold:
            reasons.append(f"drift_trigger:max_psi={max_psi}")
        alerts = report.get("alerts", [])
        critical_drift_count = len(
            [a for a in alerts if str(a.get("severity", "")).lower() == "critical" and str(a.get("type", "")).lower() == "drift"]
        )
        if critical_drift_count > max_critical:
            reasons.append(f"quality_gate_trigger:critical_drift_features={critical_drift_count}>{max_critical}")

    latency_df = load_latency_events(latency_path)
    lat = summarize_latency(latency_df)
    p95 = float(lat.get("p95_total_seconds", 0.0))
    if p95 >= latency_threshold:
        reasons.append(f"latency_trigger:p95_total_seconds={p95}")

    runs = list_runs(registry_dir=registry_dir)
    train_runs = [r for r in runs if r.get("stage") == "train"]
    if train_runs:
        latest = sorted(train_runs, key=lambda x: x.get("updated_utc", x.get("created_utc", "")))[-1]
        last_ts = latest.get("updated_utc") or latest.get("created_utc")
        if last_ts and _days_since(last_ts) >= max_days:
            reasons.append(f"periodic_trigger:days_since_last_train>={max_days}")
    else:
        reasons.append("periodic_trigger:no_train_run_found")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-driven retraining orchestrator")
    parser.add_argument("--policy", default="configs/monitoring/retrain_policy.yaml")
    parser.add_argument("--model", default=None, help="Optional model filter for train pipeline")
    parser.add_argument("--execute", action="store_true", help="Actually execute training when triggered")
    args = parser.parse_args()

    policy = _load_policy(args.policy)
    reasons = _should_retrain(policy)
    should = len(reasons) > 0

    if not should:
        print("RETRAIN_SKIPPED", {"reasons": reasons})
        return

    if not args.execute:
        print("RETRAIN_TRIGGERED_DRY_RUN", {"reasons": reasons})
        return

    train_cfg_path = policy.get("train_config_path", "configs/training/final_train.yaml")
    train_raw = load_config(train_cfg_path)
    train_cfg = flatten_train_config(train_raw)
    summary = run_train_pipeline(train_cfg, model_name=args.model)

    if policy.get("canary_after_train", True):
        run_id = summary.get("run_id")
        if run_id:
            promote_run(run_id=run_id, target_stage=policy.get("canary_stage", "canary"), registry_dir=policy.get("registry_dir", "experiments/registry"))

    print("RETRAIN_OK", {"reasons": reasons, "summary": summary})


if __name__ == "__main__":
    main()
