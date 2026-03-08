from typing import Any, Dict, List


class ConfigValidationError(ValueError):
    pass


def _require(cfg: Dict[str, Any], key: str, kind: Any = None) -> None:
    if key not in cfg:
        raise ConfigValidationError(f"Missing required config key: '{key}'")
    if kind is not None and not isinstance(cfg[key], kind):
        raise ConfigValidationError(
            f"Config key '{key}' must be {kind}, got {type(cfg[key])}"
        )


def validate_train_config(cfg: Dict[str, Any]) -> None:
    if "data_path" in cfg:
        _require(cfg, "data_path", str)
    if "target_col" in cfg:
        _require(cfg, "target_col", str)
    if "text_col" in cfg:
        _require(cfg, "text_col", str)
    if "sample_frac" in cfg:
        frac = float(cfg["sample_frac"])
        if frac <= 0 or frac > 1:
            raise ConfigValidationError("sample_frac must be in (0, 1].")
    if "n_splits" in cfg and int(cfg["n_splits"]) < 2:
        raise ConfigValidationError("n_splits must be >= 2.")


def validate_inference_config(cfg: Dict[str, Any]) -> None:
    _require(cfg, "input_path", str)
    _require(cfg, "output_path", str)
    _require(cfg, "text_col", str)
    _require(cfg, "id_col", str)


def validate_feature_config(cfg: Dict[str, Any]) -> None:
    if "data_path" in cfg:
        _require(cfg, "data_path", str)


def validate_config_for_command(command: str, cfg: Dict[str, Any]) -> None:
    cmd = command.lower().strip()
    if cmd == "train":
        validate_train_config(cfg.get("training", cfg))
        return
    if cmd == "inference":
        validate_inference_config(cfg.get("inference", cfg))
        return
    if cmd == "features":
        validate_feature_config(cfg.get("features", cfg))
        return


def flatten_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = dict(config.get("training", config))
    cv_cfg = config.get("cv", {})
    if "n_splits" not in train_cfg and isinstance(cv_cfg, dict):
        train_cfg["n_splits"] = cv_cfg.get("n_splits", 5)
        train_cfg["random_state"] = cv_cfg.get("random_state", 42)
        train_cfg["stratify"] = cv_cfg.get("stratify", False)

    trainer_cfg = train_cfg.get("trainer", {}) if isinstance(train_cfg.get("trainer", {}), dict) else {}
    if "n_splits" in trainer_cfg:
        train_cfg["n_splits"] = trainer_cfg["n_splits"]
    if "random_state" in trainer_cfg:
        train_cfg["random_state"] = trainer_cfg["random_state"]
    if "stratify" in trainer_cfg:
        train_cfg["stratify"] = trainer_cfg["stratify"]

    stacker_cfg = train_cfg.get("stacker", {}) if isinstance(train_cfg.get("stacker", {}), dict) else {}
    if "run_stacker" in stacker_cfg:
        train_cfg["run_stacker"] = stacker_cfg["run_stacker"]
    if "method" in stacker_cfg:
        train_cfg["stacker_method"] = stacker_cfg["method"]
    if "n_splits" in stacker_cfg:
        train_cfg["stacker_n_splits"] = stacker_cfg["n_splits"]
    if "params" in stacker_cfg:
        train_cfg["stacker_params"] = stacker_cfg["params"]

    model_list: List[str] = train_cfg.get("models", []) or []
    if model_list:
        train_cfg["models"] = [str(m) for m in model_list]

    return train_cfg
