"""Stage-0 zero-GPU audit: does RoboDesk carry a delayed consequence, and is a
pixel readout sensitive to it? Reads the hdf5 only -- no world model.

Decides whether branch-value credit needs a task-state utility instead of the
pixel RC reward that washed out on PushCenter.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

try:  # RoboDesk stores camera_image under a compression filter (blosc/zstd)
    import hdf5plugin  # noqa: F401
    _HAS_PLUGIN = True
except ImportError:
    _HAS_PLUGIN = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        demos = list(data.keys())[: args.episodes]
        qp, rw, st, img0, imgT = [], [], [], [], []
        pixel_ok = True
        for dm in demos:
            g = data[dm]
            qp.append(np.asarray(g["obs/qpos_objects"]))          # (35,26)
            st.append(np.asarray(g["states"]))                    # (35,76)
            rw.append(np.asarray(g["rewards"]))                   # (35,)
            if pixel_ok:
                try:
                    frames = g["obs/camera_image"]
                    img0.append(np.asarray(frames[0], dtype=np.float32) / 255.0)
                    imgT.append(np.asarray(frames[-1], dtype=np.float32) / 255.0)
                except OSError:
                    pixel_ok = False
    qp = np.stack(qp)          # (N,35,26)
    st = np.stack(st)          # (N,35,76)
    rw = np.stack(rw)          # (N,35)
    pixel_ok = pixel_ok and len(img0) == len(demos)
    if pixel_ok:
        img0 = np.stack(img0)      # (N,256,256,3)
        imgT = np.stack(imgT)

    # Reward is the ground-truth delayed task utility. Locate the "drawer" state
    # component as the state dim whose temporal trajectory best tracks the reward,
    # searching states (76-d) since qpos_objects turned out time-constant here.
    flat_r = rw.reshape(-1)
    reward_varies = float(flat_r.std()) > 1e-9
    T = rw.shape[1]
    corr = np.zeros(st.shape[2])
    if reward_varies:
        for j in range(st.shape[2]):
            col = st[:, :, j].reshape(-1)
            if col.std() > 1e-9:
                corr[j] = abs(np.corrcoef(col, flat_r)[0, 1])
    drawer = int(np.nanargmax(corr)) if reward_varies and np.nanmax(corr) > 0 else 0
    dr = st[:, :, drawer]      # (N,35) best state proxy for the drawer

    # Delayed-consequence timing is measured on the REWARD trajectory itself:
    # first step where reward rises >10% of its per-episode range toward the max.
    onset = []
    for i in range(len(rw)):
        r = rw[i]; rng = np.ptp(r) + 1e-9
        idx = np.flatnonzero(r - r[0] > 0.1 * rng)
        onset.append(int(idx[0]) if len(idx) else T)
    onset = np.asarray(onset)
    # monotonicity: fraction of steps where reward is non-decreasing
    mono = float(np.mean(np.diff(rw, axis=1) >= -1e-6))

    # pixel sensitivity: whole-frame MSE(first,last) vs the moving region.
    # Approximate the drawer region by the pixels that change most between t0 and tT.
    pixel_report = {"readable": pixel_ok, "hdf5plugin": _HAS_PLUGIN}
    if pixel_ok:
        diff = np.abs(imgT - img0).mean(-1)               # (N,256,256)
        whole_mse = ((imgT - img0) ** 2).reshape(len(img0), -1).mean(1)
        thr = np.quantile(diff.reshape(len(diff), -1), 0.95, axis=1)
        region_frac = np.mean([(diff[i] > thr[i]).mean() for i in range(len(diff))])
        pixel_report.update({
            "whole_frame_mse_t0_tT_mean": float(whole_mse.mean()),
            "whole_frame_mse_t0_tT_std": float(whole_mse.std()),
            "changed_region_frac_of_frame": float(region_frac),
        })

    report = {
        "episodes": len(demos),
        "reward": {"min": float(rw.min()), "max": float(rw.max()), "mean": float(rw.mean()),
                   "varies": bool(reward_varies),
                   "episodes_with_success": float((rw.max(1) > rw.min(1) + 1e-6).mean())},
        "reward_monotone_frac": mono,
        "reward_early_late": {"step0_mean": float(rw[:, 0].mean()),
                              "final_mean": float(rw[:, -1].mean()),
                              "mid_mean": float(rw[:, T // 2].mean())},
        "drawer_state_index": drawer,
        "drawer_reward_corr": float(corr[drawer]),
        "drawer_displacement_mean": float((dr[:, -1] - dr[:, 0]).mean()),
        "drawer_displacement_std": float((dr[:, -1] - dr[:, 0]).std()),
        "reward_onset_step": {"median": float(np.median(onset)),
                              "p25": float(np.percentile(onset, 25)),
                              "p75": float(np.percentile(onset, 75)),
                              "moved_by_end_frac": float((onset < 35).mean())},
        "pixel": pixel_report,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as h:
            json.dump(report, h, indent=2)
        print(f"saved {args.out}")
    print("STAGE0_ROBODESK_READOUT_OK")


if __name__ == "__main__":
    main()
