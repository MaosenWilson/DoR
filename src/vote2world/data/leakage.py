"""Future-GT leakage guards for adaptation-only code paths."""

from __future__ import annotations

from collections.abc import Mapping


FORBIDDEN_ADAPTATION_KEYS = frozenset(
    {
        "target_frame",
        "future_frame",
        "ground_truth",
        "label",
        "labels",
        "next_observation",
        "next_frame",
        "target_tokens",
        "future_gt",
    }
)


class FutureGTLeakageError(RuntimeError):
    """Raised when GT is passed into an adaptation path."""


def assert_no_future_gt(payload: Mapping[str, object], *, path_name: str = "adaptation") -> None:
    leaked = sorted(FORBIDDEN_ADAPTATION_KEYS.intersection(payload.keys()))
    if leaked:
        raise FutureGTLeakageError(
            f"Future-GT leakage detected in adaptation path. path={path_name} keys={leaked}"
        )

