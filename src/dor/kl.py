"""KL estimators used by the lean GRPO training loops."""

import torch


def sampled_kl_penalty(logp, ref_logp, kind="low_var_kl"):
    """Return a per-token sampled KL penalty.

    ``low_var_kl`` matches the estimator used by RLVR-World's VERL video
    recipe. ``linear`` is retained only to reproduce early pilot runs.
    """
    if kind == "low_var_kl":
        log_ratio_ref_over_policy = ref_logp - logp
        penalty = (
            torch.exp(log_ratio_ref_over_policy)
            - log_ratio_ref_over_policy
            - 1.0
        )
        return torch.clamp(penalty, min=-10.0, max=10.0)
    if kind == "linear":
        return logp - ref_logp
    raise ValueError(f"unknown KL penalty kind: {kind!r}")
