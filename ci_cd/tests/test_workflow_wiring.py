from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def _load_workflow(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _python_script_references(payload: object) -> list[str]:
    refs: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "run" and isinstance(value, str):
                refs.extend(re.findall(r"python\s+((?:scripts/|main\.py)[^\s\\\"]+)", value))
            else:
                refs.extend(_python_script_references(value))
    elif isinstance(payload, list):
        for item in payload:
            refs.extend(_python_script_references(item))
    return refs


def test_workflows_parse_cleanly() -> None:
    for workflow_path in WORKFLOWS_DIR.glob("*.yml"):
        payload = _load_workflow(workflow_path)
        assert isinstance(payload, dict), f"{workflow_path} did not parse to a workflow mapping"
        assert payload.get("jobs"), f"{workflow_path} has no jobs"


def test_workflow_python_script_references_exist() -> None:
    missing: list[str] = []
    for workflow_path in WORKFLOWS_DIR.glob("*.yml"):
        for ref in _python_script_references(_load_workflow(workflow_path)):
            target = ROOT / ref
            if ref == "main.py":
                target = ROOT / "main.py"
            if not target.exists():
                missing.append(f"{workflow_path.name}:{ref}")

    assert not missing, f"Missing workflow script references: {missing}"


def test_training_workflow_uses_canonical_local_run_outputs() -> None:
    training_workflow = (WORKFLOWS_DIR / "training.yml").read_text(encoding="utf-8")
    assert "experiments/runs/" in training_workflow
    assert 'echo "run_id=$LOCAL_RUN_ID" >> $GITHUB_OUTPUT' in training_workflow


def test_promotion_workflow_uses_valid_deployment_manifest_args() -> None:
    promote_workflow = (WORKFLOWS_DIR / "promote.yml").read_text(encoding="utf-8")
    assert "--strategy promotion" in promote_workflow
    assert "--stage" not in promote_workflow
