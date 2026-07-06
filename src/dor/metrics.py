"""Held-out GT quality metrics and rank-correlation helpers."""
import numpy as np
import torch


class Metrics:
    """Perceptual / pixel metrics against the ground-truth next frame."""

    def __init__(self, device, lpips_net="vgg"):
        import lpips as lpips_lib

        # lpips_net="vgg" matches RLVR-World's reward/eval LPIPS backbone (faithful A0
        # baseline). The earlier code used "alex"; vgg is heavier but comparable to the
        # published numbers.
        self.device = device
        self.lpips = lpips_lib.LPIPS(net=lpips_net).to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self._ssim = None

    @torch.no_grad()
    def eval_batch(self, pred, gt):
        """pred [N,3,H,W], gt [3,H,W] or [N,3,H,W] in [0,1] -> dict of [N] arrays."""
        if gt.dim() == 3:
            gt = gt.unsqueeze(0).expand_as(pred)
        mae = (pred - gt).abs().mean(dim=(1, 2, 3))
        mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse + 1e-10)
        lp = self.lpips(pred * 2 - 1, gt * 2 - 1).reshape(-1)  # lpips expects [-1,1]
        out = dict(
            mae=mae.detach().cpu().numpy(),
            mse=mse.detach().cpu().numpy(),
            psnr=psnr.detach().cpu().numpy(),
            lpips=lp.detach().cpu().numpy(),
        )
        ssim = self._maybe_ssim(pred, gt)
        if ssim is not None:
            out["ssim"] = ssim
        return out

    @torch.no_grad()
    def _maybe_ssim(self, pred, gt):
        try:
            from piqa import SSIM
        except Exception:
            return None
        if self._ssim is None:
            self._ssim = SSIM(n_channels=3, reduction="none").to(self.device)
        return self._ssim(pred.clamp(0, 1), gt.clamp(0, 1)).reshape(-1).detach().cpu().numpy()


def spearman(a, b):
    """Spearman rho without scipy."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    za = (a - a.mean()) / (a.std() + 1e-12)
    zb = (b - b.mean()) / (b.std() + 1e-12)
    return float((za * zb).mean())
