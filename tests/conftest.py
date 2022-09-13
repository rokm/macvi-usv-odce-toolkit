import os
import json
import zipfile

import pytest


@pytest.fixture
def dataset_json_file():
    variable_name = "MACVI_USV_ODCE_TEST_DATASET_JSON"
    path = os.environ.get(variable_name, None)
    if path is None:
        pytest.skip(reason=f"{variable_name} environment variable not set.")
    if not os.path.isfile(path):
        pytest.skip(reason=f"Dataset JSON file {path!r} does not exist or is not a file.")
    return path


@pytest.fixture(scope="function")
def reference_detection_results_file(method):
    variable_name = "MACVI_USV_ODCE_TEST_REFERENCE_RESULTS_DIR"
    path = os.environ.get(variable_name, None)
    if path is None:
        pytest.skip(f"{variable_name} environment variable not set.")
    if not os.path.isdir(path):
        pytest.skip(f"Reference results directory path {path!r} does not exist or is not a directory.")

    detection_results_file = os.path.join(path, f"{method}_res.json")
    if not os.path.isfile(detection_results_file):
        pytest.skip(
            f"Reference detection results for method {method!r} are not available. "
            f"Expected filename: {detection_results_file!r}."
        )

    return detection_results_file


def _reference_evaluation_results(method):
    json_file = os.path.join(os.path.dirname(__file__), "reference-evaluation-results.json")
    if not os.path.isfile(json_file):
        pytest.skip(f"Reference evaluation results file {json_file!r} does not exist or is not a file.")
    with open(json_file, "r") as fp:
        results = json.load(fp)

    if method not in results:
        pytest.skip(f"Reference evaluation results file {json_file!r} does not contain entry for method {method!r}.")

    return results[method]


@pytest.fixture(scope="function")
def reference_evaluation_results(method):
    return _reference_evaluation_results(method)


@pytest.fixture(scope="function")
def reference_evaluation_stdout_lines(method):
    s1, s2, s3 = _reference_evaluation_results(method)

    f_s1 = s1[0]
    f_s2 = s2[0]
    f_s3 = s3[0]
    f_avg = (f_s1 + f_s2 + f_s3) / 3

    return [
        "Challenge results (F_avg, F_s1, F_s2, F_s3):",  # header
        f"{f_avg:.3f} {f_s1:.3f} {f_s2:.3f} {f_s3:.3f}",  # F_avg, F_setup1, F_setup2, F_setup3
    ]


def _sample_code_archive():
    variable_name = "MACVI_USV_ODCE_TEST_SAMPLE_CODE_ARCHIVE"
    path = os.environ.get(variable_name, None)
    if path is None:
        pytest.skip(reason=f"{variable_name} environment variable not set.")
    if not os.path.isfile(path):
        pytest.skip(reason=f"Sample code archive file {path!r} does not exist or is not a file.")
    return path


@pytest.fixture()
def sample_code_archive():
    return _sample_code_archive()


@pytest.fixture()
def sample_code_dir(tmpdir):
    archive_file = _sample_code_archive()
    unpacked_code_dir = tmpdir / "sample-source-code"
    with zipfile.ZipFile(archive_file, "r") as archive:
        archive.extractall(unpacked_code_dir)
    return str(unpacked_code_dir)
