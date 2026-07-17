"""Paths and sequence/token constants for the RT-1 single-step world model.

Sequence layout (from ivideogpt SimpleVideoProcessor + config.json):
  [320 visual tokens + 13 action tokens] x 4 context frames = 1332
  + BOS (4631)                                               = 1333  (prompt)
  then generate 320 visual tokens for the predicted next frame.
Action tokens are discretized into 256 bins and offset by +VTOK.
"""
import os

ROOT = os.environ.get("VOTE2WORLD_ROOT", "/root/autodl-tmp/vote2world")
IVG = f"{ROOT}/third_party/RLVR-World/vid_wm/ivideogpt"

# token / sequence layout
CTX = 4                         # context frames
TPF = 320                       # tokens per frame
GRID = (16, 20)                 # 16 * 20 = 320
VTOK = 4375                     # visual ids: 0..4374
ABINS = 256                     # action bins
ADIM = 13                       # action dim
BOS = 4631
EOS = 4632
BLOCK = TPF + ADIM              # 333 per context frame
PROMPT_LEN = CTX * BLOCK + 1    # 1333
# RT-1 processor order used by the official action ranges and converted npz files:
# world_vector=0:3, gripper=3, rotation_delta=4:7, terminate=7:10, base=10:13.
MOTION_DIMS = [0, 1, 2, 4, 5, 6]

# checkpoints / data
TOK_DIR = f"{ROOT}/checkpoints/tokenizer/rt1-frame-tokenizer"
BASE_DIR = f"{ROOT}/checkpoints/base/rt1-world-model-single-step-base"
RLVR_DIR = f"{ROOT}/checkpoints/rlvr/rt1-world-model-single-step-rlvr"
AR_PATH = f"{IVG}/configs/vgpt/frac_action_ranges.pth"
DATA_DIR = f"{ROOT}/data/processed/fractal20220817_data"
