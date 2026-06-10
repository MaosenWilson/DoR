import json

import pytest

from vote2world.data.action_schema import ActionSchema, ActionSchemaError, write_schema_safely


def test_action_schema_covers_13_dims():
    schema = ActionSchema.from_file()
    assert schema.action_dim == 13
    covered = []
    for field in schema.fields.values():
        covered.extend(range(field.start, field.end))
    assert sorted(covered) == list(range(13))


def test_action_schema_field_slices():
    schema = ActionSchema.from_file()
    assert schema.field_index_ranges()["world_vector"] == (10, 13)
    assert schema.field_index_ranges()["rotation_delta"] == (6, 9)
    assert schema.field_index_ranges()["gripper_closedness_action"] == (0, 1)
    assert not schema.fields["terminate_episode"].include_in_motion_magnitude


def test_confirmed_schema_not_silently_overwritten(tmp_path):
    schema = ActionSchema.from_file().raw
    schema["schema_status"] = "confirmed"
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ActionSchemaError):
        write_schema_safely(schema, path, allow_overwrite=False)

