import os
import zipfile

import pytest

from macvi_usv_odce_toolkit.__main__ import main as toolkit_main


@pytest.mark.parametrize("method", ("mrcnn",))
def test_cmd_prepare_and_unpack_submission_with_code_archive(
    method,
    dataset_json_file,
    reference_detection_results_file,
    reference_evaluation_stdout_lines,
    sample_code_archive,
    tmpdir,
    capsys,
):
    # Prepare submission
    submission_archive = str(tmpdir / "submission.zip")
    toolkit_main([
        "prepare-submission",
        dataset_json_file,
        reference_detection_results_file,
        sample_code_archive,
        "--output-file",
        submission_archive,
    ])

    # Capture stdout and stderr in order to clear them
    stdout, stderr = capsys.readouterr()

    # Expected contents
    expected_contents = {
        "detection_results.json",
        "evaluation_results.json",
        "source_code/",
        "source_code/" + os.path.basename(sample_code_archive),
    }
    expected_contents = sorted(list(expected_contents))

    # Analyze resulting submission file
    with zipfile.ZipFile(submission_archive, "r") as archive:
        archive_contents = set(archive.namelist())
    archive_contents = sorted(list(archive_contents))

    assert archive_contents == expected_contents

    # Unpack submission and validate stdout
    unpacked_submission_dir = str(tmpdir / "unpacked-submission")
    toolkit_main(["unpack-submission", submission_archive, unpacked_submission_dir])

    # Capture stdout
    stdout, stderr = capsys.readouterr()
    lines = stdout.splitlines()

    # Compare the expected output
    assert lines == reference_evaluation_stdout_lines


@pytest.mark.parametrize("method", ("mrcnn",))
def test_cmd_prepare_and_unpack_submission_with_code_dir(
    method,
    dataset_json_file,
    reference_detection_results_file,
    reference_evaluation_stdout_lines,
    sample_code_dir,
    tmpdir,
    capsys,
):
    # Prepare submission
    submission_archive = str(tmpdir / "submission.zip")
    toolkit_main([
        "prepare-submission",
        dataset_json_file,
        reference_detection_results_file,
        sample_code_dir,
        "--output-file",
        submission_archive,
    ])

    # Capture stdout and stderr in order to clear them
    stdout, stderr = capsys.readouterr()

    # Expected contents
    expected_contents = {
        "detection_results.json",
        "evaluation_results.json",
        "source_code/",
    }

    def _zip_path(path):
        return path.replace(os.sep, '/') if os.sep == '\\' else path

    def _collect_dir_entries(path, root_src_path, root_dst_path):
        rel_path = os.path.relpath(path, root_src_path)  # Relative to src root
        rel_path = os.path.normpath(os.path.join(root_dst_path, rel_path))  # Prepend dst root
        rel_path = _zip_path(rel_path)  # Replace \ with /, if necessary
        if os.path.isfile(path):
            expected_contents.add(rel_path)
        elif os.path.isdir(path):
            expected_contents.add(rel_path + "/")
            for entry in os.listdir(path):
                _collect_dir_entries(os.path.join(path, entry), root_src_path, root_dst_path)

    _collect_dir_entries(sample_code_dir, os.path.dirname(sample_code_dir), "source_code")
    expected_contents = sorted(list(expected_contents))

    # Analyze resulting submission file
    with zipfile.ZipFile(submission_archive, "r") as archive:
        archive_contents = set(archive.namelist())
    archive_contents = sorted(list(archive_contents))

    assert archive_contents == expected_contents

    # Unpack submission and validate stdout
    unpacked_submission_dir = str(tmpdir / "unpacked-submission")
    toolkit_main(["unpack-submission", submission_archive, unpacked_submission_dir])

    # Capture stdout
    stdout, stderr = capsys.readouterr()
    lines = stdout.splitlines()

    # Compare expected output
    assert lines == reference_evaluation_stdout_lines
