"""Action schema helpers for RT-1 / Vote2World inputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ACTION_SCHEMA_PATH = Path("configs/vote2world/action_schema.json")


class ActionSchemaError(ValueError):
    """Raised when an action schema is invalid or unsafe to overwrite."""


@dataclass(frozen=True)
class ActionField:
    name: str
    start: int
    end: int
    dim: int
    semantic_group: str
    include_in_model_input: bool
    include_in_motion_magnitude: bool

    def slice(self) -> slice:
        return slice(self.start, self.end)


class ActionSchema:
    """Validated field-slice schema for flattened RT-1 actions."""

    def __init__(self, raw: Mapping[str, Any]):
        self.raw = dict(raw)
        self.dataset_name = str(raw["dataset_name"])
        self.schema_status = str(raw["schema_status"])
        self.action_dim = int(raw["action_dim"])
        self.flatten_key_order = list(raw["flatten_key_order"])
        self.fields = {
            name: ActionField(
                name=name,
                start=int(spec["start"]),
                end=int(spec["end"]),
                dim=int(spec["dim"]),
                semantic_group=str(spec["semantic_group"]),
                include_in_model_input=bool(spec["include_in_model_input"]),
                include_in_motion_magnitude=bool(spec["include_in_motion_magnitude"]),
            )
            for name, spec in raw["field_slices"].items()
        }
        self.official_single_step = dict(raw.get("official_single_step", {}))
        self.validate()

    @classmethod
    def from_file(cls, path: str | Path = ACTION_SCHEMA_PATH) -> "ActionSchema":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.action_dim <= 0:
            raise ActionSchemaError("action_dim must be positive")
        if set(self.flatten_key_order) != set(self.fields):
            raise ActionSchemaError("flatten_key_order and field_slices keys differ")

        covered: list[int] = []
        for name in self.flatten_key_order:
            field = self.fields[name]
            if field.end - field.start != field.dim:
                raise ActionSchemaError(f"{name} slice length does not match dim")
            if not field.include_in_model_input:
                raise ActionSchemaError(f"{name} must remain in model input")
            covered.extend(range(field.start, field.end))
        if sorted(covered) != list(range(self.action_dim)):
            raise ActionSchemaError("field slices must exactly cover action_dim without gaps/overlap")

    def slice_action(self, action: Any, field_name: str) -> Any:
        return action[..., self.fields[field_name].slice()]

    def field_index_ranges(self) -> dict[str, tuple[int, int]]:
        return {name: (field.start, field.end) for name, field in self.fields.items()}

    def to_json(self) -> str:
        return json.dumps(self.raw, indent=2, ensure_ascii=False)


def write_schema_safely(schema: Mapping[str, Any], path: str | Path, allow_overwrite: bool) -> Path:
    """Write schema, protecting existing confirmed schemas from silent overwrite."""

    dst = Path(path)
    new_schema = ActionSchema(schema)
    if dst.exists():
        old_schema = ActionSchema.from_file(dst)
        if old_schema.schema_status == "confirmed" and not allow_overwrite:
            raise ActionSchemaError("Refusing to overwrite existing confirmed schema")
        backup = dst.with_suffix(dst.suffix + "." + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + ".bak")
        shutil.copy2(dst, backup)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_schema.to_json() + "\n", encoding="utf-8")
    return dst

