import json, glob, re, sys
out = sys.argv[1] if len(sys.argv) > 1 else "outputs/grpo_full"
arms = ["pixel", "code", "ssim", "phi", "floorpc", "multi", "mse", "floor"]
for arm in arms:
    cells = []
    for f in sorted(glob.glob(f"{out}/sweep_{arm}_gt_only_s*.json")):
        s = re.search(r"_s(\d+)", f).group(1)
        d = json.load(open(f))
        log = list((d.get("run") or d.get("runs")).values())[0]
        cells.append("s%s L=%.4f flow=%.3f" % (s, log["eval_lpips"][-1], log["eval_flow"][-1]))
    print("%-9s %s" % (arm, "  ".join(cells)))
