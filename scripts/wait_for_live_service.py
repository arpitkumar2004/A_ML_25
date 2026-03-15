import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.service_probe import try_probe_live_service


def wait_for_live_service(
    base_url: str,
    expected_run_id: str = "",
    timeout_seconds: int = 600,
    interval_seconds: int = 15,
    output: str = "",
) -> dict:
    deadline = time.time() + timeout_seconds
    last_result = None

    while time.time() < deadline:
        last_result = try_probe_live_service(
            base_url=base_url,
            expected_run_id=expected_run_id or None,
            timeout_seconds=min(interval_seconds, 10),
        )
        if last_result.get("reachable") and last_result.get("valid"):
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(json.dumps(last_result, indent=2), encoding="utf-8")
            print(json.dumps(last_result, indent=2))
            return last_result
        time.sleep(interval_seconds)

    raise TimeoutError(f"Service did not become ready before timeout. Last probe={last_result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a live serving endpoint to become ready")
    parser.add_argument("--service-base-url", required=True)
    parser.add_argument("--expected-run-id", default="")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    wait_for_live_service(
        base_url=args.service_base_url,
        expected_run_id=args.expected_run_id,
        timeout_seconds=args.timeout_seconds,
        interval_seconds=args.interval_seconds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
