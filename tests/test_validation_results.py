import math
import pytest
from rococo_evaluation_lib.validation_results import ValidationResults


@pytest.fixture
def example_results():
    return ValidationResults(
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        far=[0.3, 0.3, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.0],
        frr=[0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0]
    )


def test_save__and_load_json(example_results, tmp_path):
    save_path = tmp_path / "validation_results.json"
    
    example_results.save_json(save_path)
    loaded_results = ValidationResults.load_json(save_path)

    assert loaded_results == example_results


def test_eer():
    results_1 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        frr=[0.0, 0.1, 0.2, 0.3, 0.7, 1.0]
    )

    assert results_1.eer == pytest.approx(0.2), "Should be the first FAR after crossing FRR"

    results_2 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[0.5, 0.4, 0.4, 0.3, 0.1, 0.0],
        frr=[0.0, 0.1, 0.2, 0.3, 0.7, 1.0]
    )

    assert results_2.eer == pytest.approx(0.3), "Should be the point where FAR and FRR exactly intersect"

    results_3 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[1.0, 0.8, 0.75, 0.7, 0.65, 0.6],
        frr=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    assert math.isnan(results_3.eer), "Should be NaN when no intersection occurs"


def test_eer_threshold():
    results_1 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        frr=[0.0, 0.1, 0.2, 0.3, 0.7, 1.0]
    )

    assert results_1.eer_threshold == pytest.approx(0.6), "Should be the first FAR after crossing FRR"

    results_2 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[0.5, 0.4, 0.4, 0.3, 0.1, 0.0],
        frr=[0.0, 0.1, 0.2, 0.3, 0.7, 1.0]
    )

    assert results_2.eer_threshold == pytest.approx(0.6), "Should be the point where FAR and FRR exactly intersect"

    results_3 = ValidationResults(
        threshold=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        far=[1.0, 0.8, 0.75, 0.7, 0.65, 0.6],
        frr=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    assert math.isnan(results_3.eer_threshold), "Should be NaN when no intersection occurs"


def test_frr_at_far_zero():
    results_1 = ValidationResults(
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        far=[0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        frr=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    )

    assert results_1.frr_at_far_zero == pytest.approx(0.4), "FRR is 0.4 when FAR is 0.0"

    results_2 = ValidationResults(
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        far=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        frr=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    )

    assert math.isnan(results_2.frr_at_far_zero), "Should be NaN when FAR never reaches zero"


def test_frr_at_far_zero_threshold():
    results_1 = ValidationResults(
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        far=[0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        frr=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    )

    assert results_1.frr_at_far_zero_threshold == pytest.approx(0.3), "Threshold is 0.3 when FAR is 0.0"

    results_2 = ValidationResults(
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        far=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        frr=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    )

    assert math.isnan(results_2.frr_at_far_zero_threshold), "Should be NaN when FAR never reaches zero"


def test_validation_results_initialization_lengths_not_equal():
    with pytest.raises(ValueError, match="Threshold, FAR, and FRR lists must have the same length."):
        ValidationResults(threshold=[0.0, 0.1], far=[0.1], frr=[0.2])


def test_validation_results_initialization_values_out_of_range():
    with pytest.raises(ValueError, match="FAR and FRR values must be between 0 and 1."):
        ValidationResults(threshold=[0.0, 0.1], far=[-0.1, 0.2], frr=[0.2, 1.1])