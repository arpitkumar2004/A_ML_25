import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_deployment_package import create_deployment_package
from src.utils.io import IO


def _copy_required_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _write_text_file(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_space_readme(
    path: Path,
    space_title: str,
    space_emoji: str,
    color_from: str,
    color_to: str,
    license_name: str,
    port: int,
    run_id: str,
    service_image: str,
    space_repo_id: str,
) -> None:
    body = f"""---
title: {space_title}
emoji: {space_emoji}
colorFrom: {color_from}
colorTo: {color_to}
sdk: docker
app_port: {port}
license: {license_name}
---

# {space_title}

This Space serves the `A_ML_25` FastAPI inference app as a Docker Space.

## Deployed Model

- Space repo: `{space_repo_id}`
- Canonical run ID: `{run_id}`
- Serving image tag: `{service_image}`
- Runtime bundle path inside the Space: `/opt/model-bundle`

## Runtime Endpoints

- `/`
- `/healthz`
- `/readyz`
- `/service/info`
- `/metrics/json`
- `/v1/predict`

This package was generated from the main project repo using the Hugging Face Space packaging flow.
"""
    _write_text_file(path, body)


def _write_space_dockerfile(path: Path, port: int) -> None:
    dockerfile = f"""FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PORT={port} \\
    HOST=0.0.0.0 \\
    MODEL_BUNDLE_PATH=/opt/model-bundle \\
    MODEL_RUN_ID= \\
    REGISTRY_DIR=/app/registry \\
    UVICORN_WORKERS=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \\
    pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY frontend /app/frontend
COPY main.py /app/main.py
COPY start-serving.sh /app/start-serving.sh
COPY model-bundle /opt/model-bundle
COPY metadata /app/deployment-metadata

RUN chmod +x /app/start-serving.sh

EXPOSE {port}

CMD ["/app/start-serving.sh"]
"""
    _write_text_file(path, dockerfile)


def _write_start_script(path: Path, port: int) -> None:
    script = f"""#!/usr/bin/env sh
set -eu

if [ -f /app/deployment-metadata/service.env ]; then
    set -a
    . /app/deployment-metadata/service.env
    set +a
fi

PORT="${{PORT:-{port}}}"
HOST="${{HOST:-0.0.0.0}}"
UVICORN_WORKERS="${{UVICORN_WORKERS:-1}}"

exec python -m uvicorn src.serving.app:app --host "$HOST" --port "$PORT" --workers "$UVICORN_WORKERS"
"""
    _write_text_file(path, script)


def create_hf_space_package(
    run_id: str,
    output_dir: str,
    registry_dir: str = "experiments/registry",
    service_image: str = "",
    port: int = 7860,
    space_repo_id: str = "arpitkumariitkgp/aml25",
    space_title: str = "AML25",
    space_emoji: str = "📦",
    color_from: str = "blue",
    color_to: str = "green",
    license_name: str = "mit",
) -> Dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_release = create_deployment_package(
            run_id=run_id,
            output_dir=os.path.join(tmpdir, "serving-release"),
            registry_dir=registry_dir,
            service_image=service_image,
            port=port,
        )

        _copy_required_tree(ROOT / "src", output_root / "src")
        _copy_required_tree(ROOT / "frontend", output_root / "frontend")
        shutil.copy2(ROOT / "main.py", output_root / "main.py")
        shutil.copy2(ROOT / "requirements.txt", output_root / "requirements.txt")
        _copy_required_tree(Path(base_release["bundle_output_dir"]), output_root / "model-bundle")
        _copy_required_tree(Path(base_release["metadata_dir"]), output_root / "metadata")

    _write_space_readme(
        path=output_root / "README.md",
        space_title=space_title,
        space_emoji=space_emoji,
        color_from=color_from,
        color_to=color_to,
        license_name=license_name,
        port=port,
        run_id=run_id,
        service_image=service_image,
        space_repo_id=space_repo_id,
    )
    _write_space_dockerfile(output_root / "Dockerfile", port=port)
    _write_start_script(output_root / "start-serving.sh", port=port)

    hfignore = """__pycache__/
.pytest_cache/
*.pyc
*.pyo
*.pyd
"""
    _write_text_file(output_root / ".hfignore", hfignore)

    summary = {
        "run_id": run_id,
        "space_repo_id": space_repo_id,
        "output_dir": str(output_root),
        "dockerfile_path": str(output_root / "Dockerfile"),
        "readme_path": str(output_root / "README.md"),
        "bundle_dir": str(output_root / "model-bundle"),
        "metadata_dir": str(output_root / "metadata"),
        "service_image": service_image,
    }
    IO.save_json(summary, str(output_root / "space_package_summary.json"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Hugging Face Docker Space package from a promoted run bundle")
    parser.add_argument("--run-id", required=True, help="Canonical local run ID")
    parser.add_argument("--output-dir", required=True, help="Directory to populate with the Space package")
    parser.add_argument("--registry-dir", default="experiments/registry")
    parser.add_argument("--service-image", default="")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--space-repo-id", default="arpitkumariitkgp/aml25")
    parser.add_argument("--space-title", default="AML25")
    parser.add_argument("--space-emoji", default="📦")
    parser.add_argument("--color-from", default="blue")
    parser.add_argument("--color-to", default="green")
    parser.add_argument("--license-name", default="mit")
    args = parser.parse_args()

    create_hf_space_package(
        run_id=args.run_id,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        service_image=args.service_image,
        port=args.port,
        space_repo_id=args.space_repo_id,
        space_title=args.space_title,
        space_emoji=args.space_emoji,
        color_from=args.color_from,
        color_to=args.color_to,
        license_name=args.license_name,
    )


if __name__ == "__main__":
    main()
