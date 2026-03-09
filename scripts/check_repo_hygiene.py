import os
import subprocess
from pathlib import Path


MAX_TRACKED_FILE_MB = 10


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], text=False)
    paths = [p.decode("utf-8", errors="replace") for p in out.split(b"\x00") if p]
    return paths


def _git_diff_name_only(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return []
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _candidate_files() -> list[str]:
    """Return changed files for CI/local checks without scanning full legacy history."""
    unstaged = _git_diff_name_only(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB"]
    )
    if unstaged:
        return unstaged

    staged = _git_diff_name_only(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "--cached"]
    )
    if staged:
        return staged

    # In PR CI, prefer comparing against merge-base with the base branch.
    pr_base = os.getenv("GITHUB_BASE_REF", "").strip()

    if pr_base:
        pr_changed = _git_diff_name_only(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                f"origin/{pr_base}...HEAD",
            ]
        )
        if pr_changed:
            return pr_changed

    latest_commit = _git_diff_name_only(
        ["git", "show", "--name-only", "--pretty=format:", "--diff-filter=ACMRTUXB", "HEAD"]
    )
    if latest_commit:
        return latest_commit

    return []


def _is_in(path: str, prefix: str) -> bool:
    p = path.replace("\\", "/")
    return p.startswith(prefix)


def main() -> int:
    repo_root = Path.cwd()
    files = _candidate_files()

    if not files:
        print("REPO_HYGIENE_OK (no changed files to validate)")
        return 0

    errors: list[str] = []

    blocked_exact = {"mlflow.db"}
    blocked_prefixes = ("mlruns/", "catboost_info/")

    # In these directories, only lightweight metadata should be tracked in Git.
    dvc_only_dirs = (
        "data/raw/",
        "data/processed/",
        "experiments/models/",
        "experiments/oof/",
        "experiments/reports/",
        "experiments/submissions/",
        "experiments/monitoring/",
    )

    dvc_only_allowlist = {
        "data/raw/sample_test.csv",
        "data/raw/sample_test_out.csv",
    }

    binary_exts = {".joblib", ".pkl", ".parquet", ".onnx", ".pt", ".h5"}

    max_bytes = MAX_TRACKED_FILE_MB * 1024 * 1024

    for rel_path in files:
        norm = rel_path.replace("\\", "/")

        if norm in blocked_exact or any(_is_in(norm, p) for p in blocked_prefixes):
            errors.append(f"Blocked tracked path: {norm}")

        if any(_is_in(norm, d) for d in dvc_only_dirs):
            if norm in dvc_only_allowlist:
                pass
            elif not (norm.endswith(".dvc") or norm.endswith(".gitignore") or norm.endswith("README.md")):
                errors.append(
                    "Payload tracked directly in DVC-only directory: "
                    f"{norm} (track via .dvc pointer instead)"
                )

        if Path(norm).suffix.lower() in binary_exts and not norm.endswith(".dvc"):
            errors.append(f"Binary artifact tracked directly: {norm} (track via DVC)")

        abs_path = repo_root / rel_path
        if abs_path.exists() and abs_path.is_file():
            size = abs_path.stat().st_size
            if size > max_bytes and not norm.endswith(".dvc"):
                errors.append(
                    f"Large tracked file ({size / 1024 / 1024:.1f} MB): {norm} "
                    f"(move to DVC; threshold {MAX_TRACKED_FILE_MB} MB)"
                )

    if errors:
        print("REPO_HYGIENE_FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("REPO_HYGIENE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
