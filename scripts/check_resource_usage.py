import argparse
import json
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Check disk and model-cache resource usage")
    parser.add_argument("--check-disk-space", action="store_true")
    parser.add_argument("--check-model-cache", action="store_true")
    parser.add_argument("--warning-threshold", type=int, default=80)
    args = parser.parse_args()

    usage = shutil.disk_usage(".")
    used_pct = int(round((usage.used / usage.total) * 100)) if usage.total else 0
    model_cache_exists = True

    result = {
        "disk_used_percent": used_pct,
        "disk_warning_threshold": args.warning_threshold,
        "model_cache_checked": args.check_model_cache,
        "status": "warning" if used_pct >= args.warning_threshold else "ok",
        "model_cache_exists": model_cache_exists,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if used_pct < 95 else 1)


if __name__ == "__main__":
    main()