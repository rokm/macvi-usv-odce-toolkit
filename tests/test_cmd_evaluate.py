import os
import json

import pytest

from macvi_usv_odce_toolkit.__main__ import main as toolkit_main


@pytest.mark.parametrize("method", ("mrcnn",))
def test_cmd_evaluate_with_stdout(
    method,
    dataset_json_file,
    reference_detection_results_file,
    reference_evaluation_stdout_lines,
    capsys,
):
    # Run evaluation
    toolkit_main([
        "evaluate",
        dataset_json_file,
        reference_detection_results_file,
    ])

    # Capture stdout
    stdout, stderr = capsys.readouterr()
    lines = stdout.splitlines()

    # Compare the two expected lines in stdout
    assert lines == reference_evaluation_stdout_lines


@pytest.mark.parametrize("method", ("mrcnn",))
def test_cmd_evaluate_with_file(
    method,
    dataset_json_file,
    reference_detection_results_file,
    reference_evaluation_results,
    tmpdir,
):
    # Run evaluation
    output_file = os.path.join(tmpdir, "evaluation-results.json")
    toolkit_main([
        "evaluate",
        dataset_json_file,
        reference_detection_results_file,
        "--output-file",
        output_file,
    ])

    # Load results file
    with open(output_file, "r") as fp:
        results = json.load(fp)

    # Compare
    REL_TOLERANCE = 1e-6

    assert results["setup1"] == pytest.approx(reference_evaluation_results[0], rel=REL_TOLERANCE)
    assert results["setup2"] == pytest.approx(reference_evaluation_results[1], rel=REL_TOLERANCE)
    assert results["setup3"] == pytest.approx(reference_evaluation_results[2], rel=REL_TOLERANCE)
