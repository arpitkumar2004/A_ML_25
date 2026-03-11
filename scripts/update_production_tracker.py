import argparse

from deployment_helpers import update_production_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Update production deployment tracker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args()

    update_production_tracker(args.run_id, args.strategy, args.manifest_path)


if __name__ == "__main__":
    main()