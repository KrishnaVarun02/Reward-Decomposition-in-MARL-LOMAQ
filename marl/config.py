"""Layer the four configuration types described in the report."""

import json
from pathlib import Path


def load_config(default: Path, environment: Path, algorithm: Path, test: Path) -> dict:
    result: dict = {}
    for path in (default, environment, algorithm, test):
        layer = json.loads(path.read_text())
        if not isinstance(layer, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        result = _merge(result, layer)
    return result


def _merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
