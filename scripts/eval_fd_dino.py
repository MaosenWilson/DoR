"""Offline FD-DINOv2 / KID-DINOv2 evaluation of trained GRPO checkpoints.

Distributional / feature-level eval -- answers the advisor's "features have their own
standard (FID etc.), don't judge them with LPIPS/SSIM". For each checkpoint we load the
policy, generate K candidates per held-out window, decode to pixels, embed the decoded
frames with DINOv2 (CLS), and compare the *distribution* of generated frames to the
distribution of ground-truth next-frames via:
  - FD-DINOv2 : Frechet distance between the two feature Gaussians (FID with DINOv2).
  - KID-DINOv2: unbiased polynomial-kernel MMD^2 (robust at small sample size).
Both are SET statistics -> evaluation only, NEVER a per-sample reward (one sample's FID
is undefined). Lower is better. All arms share the same held-out windows + per-window
generation seed, so the comparison is paired.

Example:
  python scripts/eval_fd_dino.py --n 512 --K 16 --seed 0 \
    --arms pixel=outputs/grpo_v1/singles/ckpt/pixel_gt_only_s0,\
code=outputs/grpo_v1/singles/ckpt/code_gt_only_s0,base=BASE
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy import linalg
from scipy.spatial.distance import cdist

from dor.constants import CTX
from dor.episodes import get_window_tensors, list_episodes, sample_windows
from dor.generation import generate_candidates
from dor.models import load_action_ranges, load_tokenizer, load_world_model
from dor.tokenization import build_prompt, decode_tokens

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Phi:
    """DINOv2 CLS embedder. imgs in [0,1] [B,3,H,W] -> [B,D] (same as probe_phi_dino)."""

    def __init__(self, dev, name="facebook/dinov2-small"):
        from transformers import Dinov2Model
        self.m = Dinov2Model.from_pretrained(name).to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(dev), _STD.to(dev)

    @torch.no_grad()
    def __call__(self, imgs):
        x = Fnn.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.m(pixel_values=x).last_hidden_state[:, 0]  # CLS [B,D]


class Inception:
    """torchvision Inception-V3 pool3 (2048-d) embedder -- the classic FID/KID backbone."""

    def __init__(self, dev):
        import torchvision
        m = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT,
            transform_input=False, aux_logits=True)
        m.fc = torch.nn.Identity()
        self.m = m.to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(dev), _STD.to(dev)

    @torch.no_grad()
    def __call__(self, imgs):
        x = Fnn.interpolate(imgs, size=299, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.m(x)  # [B,2048] (fc replaced by Identity, eval mode -> no aux)


def _knn_radius(X, k):
    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]  # distance to k-th nearest neighbour


def prdc(R, G, k=5):
    """Precision/Recall (Kynkaanniemi 2019) + Density/Coverage (Naeem 2020).

    R real features [nR,d], G generated [nG,d]. Higher is better for all four.
    P = realism (fakes inside real manifold); R = diversity (reals covered by fakes).
    """
    dRG = cdist(R, G)                  # [nR,nG]
    rR = _knn_radius(R, k)[:, None]    # [nR,1]
    rG = _knn_radius(G, k)[None, :]    # [1,nG]
    inR = dRG <= rR                    # fake g inside real r's ball
    precision = inR.any(0).mean()
    recall = (dRG <= rG).any(1).mean()
    density = (1.0 / k) * inR.sum(0).mean()
    coverage = inR.any(1).mean()
    return float(precision), float(recall), float(density), float(coverage)


def frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + off).dot(cov2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def kid(X, Y, deg=3, c=1.0):
    """Unbiased polynomial-kernel MMD^2 (KID). X [n,d], Y [m,d]; robust at small n."""
    d = X.shape[1]
    poly = lambda A, B: (A.dot(B.T) / d + c) ** deg
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = poly(X, X), poly(Y, Y), poly(X, Y)
    sx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sx + sy - 2 * Kxy.mean())


@torch.no_grad()
def embed_gt(windows, tok, phi, dev):
    G = []
    for p, s in windows:
        frames, _ = get_window_tensors(p, s, dev)
        G.append(phi(frames[CTX].unsqueeze(0)).cpu().numpy())
    return np.concatenate(G, 0)


@torch.no_grad()
def embed_pred(ckpt, windows, tok, ar, phi, K, seed, dev):
    from transformers import AutoModelForCausalLM
    if ckpt.upper() == "BASE":
        model = load_world_model(dev, "base")
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).to(dev).eval()
    P = []
    for wi, (p, s) in enumerate(windows):
        frames, actions = get_window_tensors(p, s, dev)
        prompt = build_prompt(tok, frames, actions, ar)
        cand = generate_candidates(model, prompt, K, seed=seed * 100000 + wi)
        P.append(phi(decode_tokens(tok, cand)).cpu().numpy())
        if (wi + 1) % 50 == 0:
            print(f"    [{wi + 1}/{len(windows)}]", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(P, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="label=ckptdir,...  (use label=BASE for the pre-RL model)")
    ap.add_argument("--n", type=int, default=512, help="held-out windows (FD needs n > feature dim ~384)")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default="dino", choices=["dino", "inception"],
                    help="feature space for the distributional panel")
    ap.add_argument("--dino", default="facebook/dinov2-small")
    ap.add_argument("--k", type=int, default=5, help="k for precision/recall/density/coverage")
    ap.add_argument("--win_seed", type=int, default=12345, help="held-out window sampling seed (shared by arms)")
    args = ap.parse_args()
    dev = "cuda"

    tok = load_tokenizer(dev)
    ar = load_action_ranges(dev)
    phi = Inception(dev) if args.embedder == "inception" else Phi(dev, args.dino)
    space = "Inception" if args.embedder == "inception" else args.dino
    windows = list(sample_windows(list_episodes(), args.n, seed=args.win_seed))
    print(f"[setup] embedder={space}  windows={len(windows)}  K={args.K}  k_prdc={args.k}", flush=True)

    G = embed_gt(windows, tok, phi, dev)
    muG, covG = G.mean(0), np.cov(G, rowvar=False)
    D = G.shape[1]
    if len(G) <= D:
        print(f"[warn] nG={len(G)} <= feat dim {D}: FD rank-deficient -> read KID/PRDC, raise --n", flush=True)

    rows = []
    for spec in args.arms.split(","):
        label, ck = spec.split("=", 1)
        print(f"[arm] {label}: {ck}", flush=True)
        P = embed_pred(ck, windows, tok, ar, phi, args.K, args.seed, dev)
        fd = frechet(P.mean(0), np.cov(P, rowvar=False), muG, covG)
        ki = kid(P, G)
        prec, rec, dens, cov = prdc(G, P, k=args.k)
        rows.append((label, fd, ki, prec, rec, dens, cov))
        print(f"  -> FD={fd:.2f} KID={ki*1e3:.2f}e-3 | P={prec:.3f} R={rec:.3f} "
              f"Dens={dens:.3f} Cov={cov:.3f}  (nP={len(P)},nG={len(G)})", flush=True)

    print(f"\n=== Distributional panel @ {space}  (eval only, never a reward) ===")
    print("    FD/KID lower=better; Precision/Recall/Density/Coverage higher=better")
    print(f"  {'arm':14s} {'FD':>7s} {'KID(e-3)':>9s} {'Prec':>6s} {'Recall':>7s} {'Dens':>6s} {'Cov':>6s}")
    for label, fd, ki, prec, rec, dens, cov in rows:
        print(f"  {label:14s} {fd:7.2f} {ki*1e3:9.2f} {prec:6.3f} {rec:7.3f} {dens:6.3f} {cov:6.3f}")
    print("DIST_PANEL_OK")


if __name__ == "__main__":
    main()
