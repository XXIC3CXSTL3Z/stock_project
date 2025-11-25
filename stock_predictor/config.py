import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """Load settings from a YAML or JSON config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
    elif path.suffix.lower() == ".json":
        with path.open() as f:
            data = json.load(f)
    else:
        raise ValueError("Config file must be YAML or JSON.")

    if not isinstance(data, dict):
        raise ValueError("Config content must be a mapping.")
    return data


def merge_config(cli_args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge CLI arguments with config values (config wins when not None).
    """
    merged = dict(cli_args)
    for key, value in config.items():
        if value is not None:
            merged[key] = value
    return merged
