"""YAML configuration loader with basic validation."""
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def load_all_configs(config_dir: str | Path) -> dict[str, Any]:
    config_dir = Path(config_dir)
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        key = yaml_file.stem
        configs[key] = load_config(yaml_file)
    return configs
