from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = files("road_analytics").joinpath("default.yaml")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    return config
