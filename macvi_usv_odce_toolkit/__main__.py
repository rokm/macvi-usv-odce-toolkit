import sys
import os
import argparse
import logging
import time
import zipfile
import json

import macvi_usv_odce_toolkit


def _process_sequences_list(sequences_list_str):
    sequence_ids = set()

    # Split on comma
    for token in sequences_list_str.split(","):
        elements = token.split("-")
        if len(elements) == 1:
            sequence_ids.add(int(elements[0]))
        elif len(elements) == 2:
            for i in range(int(elements[0]), int(elements[1]) + 1):
                sequence_ids.add(i)
        else:
            logging.warning("Invalid sequence list token: %r", token)

    return sorted(list(sequence_ids))


def _perform_full_evaluation(dataset_json_file, results_json_file, sequences=None):
    results = {}

    # Setup 1
    logging.info("Evaluating Setup 1...")
    start_time = time.time()
    results['setup1'] = macvi_usv_odce_toolkit.evaluate_detection_results_setup1(
        dataset_json_file,
        results_json_file,
        sequences,
    )
    elapsed = time.time() - start_time
    logging.info("Evaluation complete in %.2f seconds!", elapsed)

    # Setup 2
    logging.info("Evaluating Setup 2...")
    start_time = time.time()
    results['setup2'] = macvi_usv_odce_toolkit.evaluate_detection_results_setup2(
        dataset_json_file,
        results_json_file,
        sequences,
    )
    elapsed = time.time() - start_time
    logging.info("Evaluation complete in %.2f seconds!", elapsed)

    # Setup 3
    logging.info("Evaluating Setup 3...")
    start_time = time.time()
    results['setup3'] = macvi_usv_odce_toolkit.evaluate_detection_results_setup3(
        dataset_json_file,
        results_json_file,
        sequences,
    )
    elapsed = time.time() - start_time
    logging.info("Evaluation complete in %.2f seconds!", elapsed)
    logging.info("")

    return results


def _display_extended_results(results):
    # Display extended results to stderr, using logging.info()
    logging.info("Results: F_all F_small F_medium F_large")
    logging.info("Setup_1: %.03f %.03f %.03f %.03f", *results['setup1'])
    logging.info("Setup_2: %.03f %.03f %.03f %.03f", *results['setup2'])
    logging.info("Setup_3: %.03f %.03f %.03f %.03f", *results['setup3'])
    logging.info("")


def _display_final_results(results):
    # Display final results to stdout
    f_s1 = results['setup1'][0]  # F_all for Setup 1
    f_s2 = results['setup2'][0]  # F_all for Setup 2
    f_s3 = results['setup3'][0]  # F_all for Setup 3
    f_avg = (f_s1 + f_s2 + f_s3) / 3

    print("Challenge results (F_avg, F_s1, F_s2, F_s3):")
    print(f"{f_avg:.03f} {f_s1:.03f} {f_s2:.03f} {f_s3:.03f}")


def _collect_to_archive(archive, path, archive_path):
    if os.path.isfile(path):
        archive.write(path, archive_path)
    elif os.path.isdir(path):
        if archive_path:
            archive.write(path, archive_path)
        for nm in sorted(os.listdir(path)):
            _collect_to_archive(archive, os.path.join(path, nm), os.path.join(archive_path, nm))


def cmd_evaluate(args):
    """
    Command handler: evaluate

    Evaluates the detection results and prints the summary to standard output.

    Parameters
    ----------
    args : argparse.Namespace
        argparse Namespace structure, obtained by argparse.ArgumentParser.parse_args().
    """
    # Collect arguments
    dataset_json_file = getattr(args, 'dataset-json-file')
    results_json_file = getattr(args, 'results-json-file')
    output_file = args.output_file
    sequences = args.sequences

    if sequences is not None:
        sequences = _process_sequences_list(sequences)

    # Display settings
    logging.info("")
    logging.info("Settings:")
    logging.info(" - mode: %r", args.command)
    logging.info(" - dataset JSON file: %r", dataset_json_file)
    logging.info(" - results JSON file: %r", results_json_file)
    logging.info(" - output file: %r", output_file)
    logging.info(" - sequence(s): %r", sequences)
    logging.info("")

    # Run the evaluation
    results = _perform_full_evaluation(dataset_json_file, results_json_file, sequences)

    # Display debug/extended results
    _display_extended_results(results)

    # Display actual (short) results
    _display_final_results(results)

    # Save
    if output_file:
        logging.info("")
        logging.info("Saving evaluation results to %r...", output_file)
        with open(output_file, "w") as fp:
            json.dump(results, fp, indent=2)

    # Done
    logging.info("")
    logging.info("Done!")


def cmd_prepare_submission(args):
    """
    Command handler: prepare-submission

    Evaluates the detection results and prepares submission archive by collecting raw detection results file, evaluation
    results file, and source code.

    Parameters
    ----------
    args : argparse.Namespace
        argparse Namespace structure, obtained by argparse.ArgumentParser.parse_args().
    """
    # Collect arguments
    dataset_json_file = getattr(args, 'dataset-json-file')
    results_json_file = getattr(args, 'results-json-file')
    source_code_path = getattr(args, 'source-code-path')
    output_file = args.output_file

    # Display settings
    logging.info("")
    logging.info("Settings:")
    logging.info(" - mode: %r", args.command)
    logging.info(" - dataset JSON file: %r", dataset_json_file)
    logging.info(" - results JSON file: %r", results_json_file)
    logging.info(" - source code path: %r", source_code_path)
    logging.info(" - output file: %r", output_file)
    logging.info("")

    # Validate source code file/directory
    if not os.path.isfile(source_code_path) and not os.path.isdir(source_code_path):
        logging.error("Invalid source code path %r: not a file or directory!", source_code_path)
        sys.exit(-1)

    # Run the evaluation (always on all sequences)
    results = _perform_full_evaluation(dataset_json_file, results_json_file)

    # Display debug/extended results
    _display_extended_results(results)

    # Display actual (short) results
    _display_final_results(results)

    # Prepare submission archive
    logging.info("")
    logging.info("Preparing submission archive %r...", output_file)

    with zipfile.ZipFile(output_file, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Collect raw detection results JSON as detection_results.json
        logging.info("Collecting raw results file %r...", results_json_file)
        archive.write(results_json_file, "detection_results.json")

        # Collect evaluation results JSON as evaluation_results.json
        # (dump to JSON string and write to archive directly, without temporary file)
        logging.info("Collecting evaluation results file...")
        evaluation_json_data = json.dumps(results, indent=2)
        archive.writestr("evaluation_results.json", evaluation_json_data)

        # Collect source code into source_code directory
        logging.info("Collecting source code from %r...", source_code_path)
        archive.writestr("source_code/", "")  # Create empty directory
        _collect_to_archive(
            archive,
            source_code_path,
            os.path.join("source_code", os.path.basename(source_code_path)),
        )

    # Done
    logging.info("")
    logging.info("Done!")


def cmd_unpack_submission(args):
    """
    Command handler: unpack-submission

    Unpacks the submission archive to target location, and prints the evaluation results to standard output.
    Can optionally perform re-evaluation of raw results using local copy of toolkit and dataset.

    Parameters
    ----------
    args : argparse.Namespace
        argparse Namespace structure, obtained by argparse.ArgumentParser.parse_args().
    """
    # Collect arguments
    submission_file = getattr(args, 'submission-file')
    target_path = getattr(args, 'target-path')
    dataset_json_file = args.dataset_json_file

    # Display settings
    logging.info("")
    logging.info("Settings:")
    logging.info(" - mode: %r", args.command)
    logging.info(" - submission archive: %r", submission_file)
    logging.info(" - target path: %r", target_path)
    logging.info(" - dataset JSON file (local re-evaluation): %r", dataset_json_file)
    logging.info("")

    # Unpack submission
    logging.info("Unpacking submission...")
    os.makedirs(target_path, exist_ok=False)  # Raise exception if trying to unpack into existing directory
    with zipfile.ZipFile(submission_file, mode="r") as archive:
        archive.extractall(target_path)
    logging.info("")

    # Local re-evaluation?
    if dataset_json_file:
        results_json_file = os.path.join(target_path, "detection_results.json")
        logging.info("Performing local re-evaluation of raw results...")
        results = _perform_full_evaluation(dataset_json_file, results_json_file)
    else:
        evaluation_json_file = os.path.join(target_path, "evaluation_results.json")
        logging.info("Using submitted evaluation results...")
        with open(evaluation_json_file, 'r') as fp:
            results = json.load(fp)
    logging.info("")

    # Display debug/extended results
    _display_extended_results(results)

    # Display actual (short) results
    _display_final_results(results)


def main(args=None):
    """
    Entry-point function.

    Parameters
    ----------
    args : iterable or None
        List of command-line arguments without application name. If not provided, sys.argv[1:] is used.
    """

    if args is None:
        args = sys.argv[1:]

    # *** Basic logging setup ***
    logging.basicConfig(
        level=logging.INFO,
        format="{message}",
        style="{",
    )

    # *** Command-line parser ***
    parser = argparse.ArgumentParser(
        prog="macvi_usv_odce_tool",
        description="MaCVi USV Obstacle Detection Challenge Evaluation Toolkit",
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0',
    )

    # Sub-commands
    subparsers = parser.add_subparsers(
        title="valid commands",
        metavar="command",
        help="command description",
        required=True,
    )

    # Command: evaluate
    subparser = subparsers.add_parser(
        "evaluate",
        aliases=["e"],
        help="Evaluate the results.",
    )
    subparser.set_defaults(
        command="evaluate",
        command_function=cmd_evaluate,
    )
    subparser.add_argument(
        "dataset-json-file",
        type=str,
        help="Full path to the MODS dataset JSON file (mods.json).",
    )
    subparser.add_argument(
        "results-json-file",
        type=str,
        help="Full path to the JSON file with detection results.",
    )
    subparser.add_argument(
        "--output-file",
        type=str,
        metavar="FILENAME",
        help="Store evaluation results in a JSON file in addition to displaying them in console.",
    )
    subparser.add_argument(
        "--sequences",
        type=str,
        metavar="SEQUENCE_LIST",
        help="Optional comma-separated list of sequence IDs to use during evaluation (default: use all sequences).",
    )

    # Command: prepare-submission
    subparser = subparsers.add_parser(
        "prepare-submission",
        aliases=["s"],
        help="Evaluate the results and prepare archive for submission.",
    )
    subparser.set_defaults(
        command="prepare-submission",
        command_function=cmd_prepare_submission,
    )
    subparser.add_argument(
        "dataset-json-file",
        type=str,
        help="Full path to the MODS dataset JSON file (mods.json).",
    )
    subparser.add_argument(
        "results-json-file",
        type=str,
        help="Full path to the JSON file with detection results.",
    )
    subparser.add_argument(
        "source-code-path",
        type=str,
        help="Full path to file or directory containing the source code to be included with submission.",
    )
    subparser.add_argument(
        "--output-file",
        type=str,
        default="submission.zip",
        help="Name of the generated archive for submission.",
    )

    # Command: unpack-submission
    subparser = subparsers.add_parser(
        "unpack-submission",
        aliases=["u"],
        help="Unpack the submission archive.",
    )
    subparser.set_defaults(
        command="unpack-submission",
        command_function=cmd_unpack_submission,
    )
    subparser.add_argument(
        "submission-file",
        type=str,
        help="Full path to the submission archive to unpack.",
    )
    subparser.add_argument(
        "target-path",
        type=str,
        help="Full path to directory into which submission archive is to be unpacked.",
    )
    subparser.add_argument(
        "--dataset-json-file",
        type=str,
        metavar="FILENAME",
        help="Full path to the MODS dataset JSON file (mods.json) for local result re-evaluation.",
    )

    # *** Parse command-line arguments ***
    args = parser.parse_args(args)

    # *** Run the command ***
    logging.info("MaCVi USV Obstacle Detection Challenge Evaluation Toolkit")
    args.command_function(args)


if __name__ == '__main__':
    main()
