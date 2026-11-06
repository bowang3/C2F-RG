"""Small configuration helpers for restoring the released ClinFusion layout."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


class ModelBuildArgs(SimpleNamespace):
    """Attributes expected by the original ClinFusion encoder builder."""

    def __init__(self, **kwargs):
        super().__init__(use_fast_tokenizer=True, **kwargs)

    def update(self, values: dict) -> None:
        self.__dict__.update(values)


def get_config_values(path: str, keys: list[str], *, strict: bool = True) -> dict:
    file = Path(path)
    if file.is_dir():
        file /= "config.json"
    with file.open(encoding="utf-8") as handle:
        config = json.load(handle)
    result = {}
    for key in keys:
        value = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                if strict:
                    raise KeyError(f"Missing {key!r} in {file}")
                value = None
                break
            value = value[part]
        result[key] = value
    return result
