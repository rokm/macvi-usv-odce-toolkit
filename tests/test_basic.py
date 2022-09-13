import pytest

import macvi_usv_odce_toolkit as toolkit


@pytest.mark.parametrize("method", ("mrcnn", "yolo", "fcos", "ssd"))
def test_evaluation(dataset_json_file, reference_detection_results_file, reference_evaluation_results):
    REL_TOLERANCE = 1e-6

    # Evaluate Setup #1
    evaluation_results = toolkit.evaluate_detection_results_setup1(
        dataset_json_file,
        reference_detection_results_file,
    )
    assert evaluation_results == pytest.approx(reference_evaluation_results[0], rel=REL_TOLERANCE)

    # Evaluate Setup #2
    evaluation_results = toolkit.evaluate_detection_results_setup2(
        dataset_json_file,
        reference_detection_results_file,
    )
    assert evaluation_results == pytest.approx(reference_evaluation_results[1], rel=REL_TOLERANCE)

    # Evaluate Setup #3
    evaluation_results = toolkit.evaluate_detection_results_setup3(
        dataset_json_file,
        reference_detection_results_file,
    )
    assert evaluation_results == pytest.approx(reference_evaluation_results[2], rel=REL_TOLERANCE)
