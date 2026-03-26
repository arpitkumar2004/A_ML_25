import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.service_probe import try_probe_live_prediction


def live_service_smoke_test(
    service_base_url: str,
    expected_run_id: str = "",
    service_timeout_seconds: float = 15.0,
    prediction_timeout_seconds: float = 120.0,
    retries: int = 3,
    retry_delay_seconds: float = 20.0,
    output: str = "",
) -> dict:
    attempts = max(int(retries), 1)
    result = None

    for attempt in range(1, attempts + 1):
        result = try_probe_live_prediction(
            base_url=service_base_url,
            expected_run_id=expected_run_id or None,
            service_timeout_seconds=service_timeout_seconds,
            prediction_timeout_seconds=prediction_timeout_seconds,
        )
        if result.get("valid"):
            break
        if attempt < attempts:
            time.sleep(max(float(retry_delay_seconds), 0.0))

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live prediction smoke test against the deployed service")
    parser.add_argument("--service-base-url", required=True)
    parser.add_argument("--expected-run-id", default="")
    parser.add_argument("--service-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--prediction-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=20.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = live_service_smoke_test(
        service_base_url=args.service_base_url,
        expected_run_id=args.expected_run_id,
        service_timeout_seconds=args.service_timeout_seconds,
        prediction_timeout_seconds=args.prediction_timeout_seconds,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        output=args.output,
    )
    raise SystemExit(0 if result.get("valid") else 1)


if __name__ == "__main__":
    main()
