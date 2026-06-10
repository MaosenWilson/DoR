# Vote2World Input Pipeline Report

## Schema

- schema status: `provisional`
- action dim: `13`
- action schema file: `configs/vote2world/action_schema.json`

## Dataset Validation

- episodes loaded: `1`
- adaptation windows: `2`
- evaluation windows: `2`
- adaptation keys: `['context_actions', 'context_frames', 'episode_id', 'sample_id', 'start_index']`
- evaluation keys: `['context_actions', 'context_frames', 'episode_id', 'sample_id', 'start_index', 'target_frame']`
- context_frames shape: `[4, 3, 8, 10]`
- context_actions shape: `[4, 13]`
- current frame is `context_frames[-1]`: `True`
- current action is `context_actions[-1]`: `True`

## Token Layout

- visual tokens per frame: `320`
- action tokens per step: `13`
- 4-step history length: `1332`
- BOS position: `1332`
- generation input length: `1333`
- generation output length: `321`
- decoder output shape: `[1, 3, 256, 320]`

## Future-GT Isolation

- leakage guard: `pass`
- adaptation path returns no `target_frame`
- evaluation path includes `target_frame`

## Blockers

- action_schema.json is still provisional; run official converter/TFDS schema audit to confirm key order.

READY_FOR_CANDIDATE_SAMPLING = NO
