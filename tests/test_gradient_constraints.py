import pytest
import torch

from dor.gradient_constraints import (
    accumulate_parameter_gradients,
    correction_projection_statistics,
    project_to_primary_progress,
    projection_statistics,
)


def test_conflicting_auxiliary_is_projected_to_anchor_boundary():
    stats = projection_statistics(dot=-2.0, primary_norm_sq=4.0, auxiliary_norm_sq=5.0)
    assert stats["conflict"] is True
    assert stats["projection_coefficient"] == pytest.approx(-0.5)
    assert stats["projected_auxiliary_norm"] == pytest.approx(2.0)
    assert stats["combined_anchor_ratio"] == pytest.approx(1.0)


def test_aligned_auxiliary_is_only_norm_capped():
    stats = projection_statistics(dot=2.0, primary_norm_sq=1.0, auxiliary_norm_sq=4.0)
    assert stats["conflict"] is False
    assert stats["projection_coefficient"] == 0.0
    assert stats["auxiliary_cap"] == pytest.approx(0.5)
    assert stats["retained_auxiliary_ratio"] == pytest.approx(0.5)
    assert stats["combined_anchor_ratio"] == pytest.approx(2.0)


def test_zero_gradient_is_rejected():
    with pytest.raises(ValueError):
        projection_statistics(dot=0.0, primary_norm_sq=0.0, auxiliary_norm_sq=1.0)


def test_correction_audit_recovers_conflicting_increment_without_materializing_it():
    # g_primary=(1,0), g_corrected=(0.5,1), hence correction=(-0.5,1).
    stats = correction_projection_statistics(
        primary_norm_sq=1.0,
        corrected_norm_sq=1.25,
        primary_corrected_dot=0.5,
    )
    assert stats["conflict"] is True
    assert stats["auxiliary_norm"] == pytest.approx(5 ** 0.5 / 2)
    assert stats["projected_auxiliary_norm"] == pytest.approx(1.0)
    assert stats["corrected_anchor_ratio"] == pytest.approx(0.5)
    assert stats["combined_anchor_ratio"] == pytest.approx(1.0)
    assert stats["safe_primary_cosine"] > stats["corrected_primary_cosine"]


def test_aligned_correction_is_not_projected():
    # g_primary=(1,0), g_corrected=(2,1), correction=(1,1).
    stats = correction_projection_statistics(
        primary_norm_sq=1.0,
        corrected_norm_sq=5.0,
        primary_corrected_dot=2.0,
    )
    assert stats["conflict"] is False
    assert stats["projection_coefficient"] == 0.0
    assert stats["combined_anchor_ratio"] > 1.0


def test_identical_corrected_gradient_is_a_clean_noop():
    stats = correction_projection_statistics(
        primary_norm_sq=4.0,
        corrected_norm_sq=4.0,
        primary_corrected_dot=4.0,
    )
    assert stats["auxiliary_norm"] == 0.0
    assert stats["retained_auxiliary_ratio"] == 0.0
    assert stats["safe_primary_cosine"] == pytest.approx(1.0)
    assert stats["corrected_primary_cosine"] == pytest.approx(1.0)


def test_primary_progress_projection_is_closed_form_halfspace_projection():
    primary = (torch.tensor([1.0, 0.0]),)
    preferred = (torch.tensor([-2.0, 3.0]),)
    projected, stats = project_to_primary_progress(primary, preferred)

    torch.testing.assert_close(projected[0], torch.tensor([1.0, 3.0]))
    assert stats["constraint_active"] is True
    assert stats["coefficient"] == pytest.approx(3.0)
    assert stats["preferred_progress_ratio"] == pytest.approx(-2.0)
    assert stats["projected_progress_ratio"] == pytest.approx(1.0)


def test_primary_progress_projection_keeps_feasible_preferred_gradient():
    primary = (torch.tensor([1.0, 0.0]),)
    preferred = (torch.tensor([1.5, 2.0]),)
    projected, stats = project_to_primary_progress(primary, preferred)

    torch.testing.assert_close(projected[0], preferred[0])
    assert stats["constraint_active"] is False
    assert stats["coefficient"] == pytest.approx(0.0)
    assert stats["projected_progress_ratio"] == pytest.approx(1.5)


def test_accumulate_parameter_gradients_adds_scaled_detached_values():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    accumulate_parameter_gradients(
        (parameter,), (torch.tensor([2.0, 4.0]),), scale=0.5
    )
    accumulate_parameter_gradients(
        (parameter,), (torch.tensor([1.0, -1.0]),), scale=1.0
    )
    torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 1.0]))
