import argparse

from rollback_deployment import rollback_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto rollback wrapper for health-check workflow")
    parser.add_argument("--steps-back", type=int, default=2)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--approval-reason", required=True)
    args = parser.parse_args()

    rollback_deployment(to_previous_production=True, reason=f"{args.reason} | {args.approval_reason}")


if __name__ == "__main__":
    main()