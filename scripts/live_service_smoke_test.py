import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.service_probe import try_probe_live_prediction


def live_service_smoke_test(
    service_base_url: str,
    expected_run_id: str = "",
    output: str = "",
) -> dict:
    result = try_probe_live_prediction(
        base_url=service_base_url,
        expected_run_id=expected_run_id or None,
    )

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
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = live_service_smoke_test(
        service_base_url=args.service_base_url,
        expected_run_id=args.expected_run_id,
        output=args.output,
    )
    raise SystemExit(0 if result.get("valid") else 1)


if __name__ == "__main__":
    main()
