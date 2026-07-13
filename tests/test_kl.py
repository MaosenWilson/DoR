import torch

from dor.kl import sampled_kl_penalty


def test_low_var_kl_matches_verl_formula():
    logp = torch.tensor([-2.0, -0.7, -3.1])
    ref = torch.tensor([-1.6, -1.0, -2.9])
    delta = ref - logp
    expected = torch.exp(delta) - delta - 1.0
    torch.testing.assert_close(
        sampled_kl_penalty(logp, ref, "low_var_kl"), expected
    )


def test_low_var_kl_gradient_pulls_sampled_logp_toward_reference():
    logp = torch.tensor([-2.0, -0.5], requires_grad=True)
    ref = torch.tensor([-1.0, -1.5])
    sampled_kl_penalty(logp, ref, "low_var_kl").sum().backward()

    # Gradient descent increases an underweighted token and decreases an
    # overweighted token, moving both sampled log-probabilities toward ref.
    assert logp.grad[0] < 0
    assert logp.grad[1] > 0


def test_linear_mode_is_kept_for_reproducing_legacy_runs():
    logp = torch.tensor([-2.0, -0.5])
    ref = torch.tensor([-1.0, -1.5])
    torch.testing.assert_close(
        sampled_kl_penalty(logp, ref, "linear"), logp - ref
    )
