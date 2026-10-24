"""Configuration loading and device selection helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {"horizontal_fov_deg": 70.0, "focal_length_px": None},
    "object_widths_m": {"cone": 0.30, "barrier": 1.50, "stop_sign": 0.75},
    "inference": {"confidence": 0.25, "image_size": 640, "device": "auto"},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and merge it over safe project defaults."""
    config: dict[str, Any] = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in DEFAULT_CONFIG.items()
    }
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    for section, values in loaded.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


def normalize_class_name(name: str) -> str:
    """Normalize labels such as 'Stop Sign' to config key 'stop_sign'."""
    return "_".join(name.strip().lower().replace("-", " ").split())
