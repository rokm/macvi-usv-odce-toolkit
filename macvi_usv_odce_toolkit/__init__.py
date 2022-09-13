from . import evaluation


def evaluate_detection_results_setup1(dataset_json_file, results_json_file, sequences=None):
    """
    Evaluation detection results using Setup 1 from Section VI-A in Bovcon et al. paper.

    In Setup 1, standard detection evaluation protocol is used, but only within the part
    of the image whete the obstacles can appear; i.e., using the sea-edge based mask.

    This function is a helper wrapper for evaluation.evaluate_detection_results() function.
    """
    return evaluation.evaluate_detection_results(
        dataset_json_file,
        results_json_file,
        sequences,
        # Setup 1: sea-edge based mask, use class information
        mode='edge',
    )


def evaluate_detection_results_setup2(dataset_json_file, results_json_file, sequences=None):
    """
    Evaluation detection results using Setup 2 from Section VI-A in Bovcon et al. paper.

    Setup 2 is similar to Setup 1, except that class information is ignored. Therefore,
    it evaluates only obstacle detection without class identification, which is crucial
    for path planning and collision avoidance.

    This function is a helper wrapper for evaluation.evaluate_detection_results() function.
    """
    return evaluation.evaluate_detection_results(
        dataset_json_file,
        results_json_file,
        sequences,
        # Setup 2: sea-edge based mask, ignore class information
        mode='edge',
        ignore_class=True,
    )


def evaluate_detection_results_setup3(dataset_json_file, results_json_file, sequences=None):
    """
    Evaluation detection results using Setup 3 from Section VI-A in Bovcon et al. paper.

    Setup 3 follows Setup 2, but analyzes performance only within the USV danger zone
    (Section V-C in Bovcon et al. paper), i.e., using the danger-zone based mask.

    This function is a helper wrapper for evaluation.evaluate_detection_results() function.
    """
    return evaluation.evaluate_detection_results(
        dataset_json_file,
        results_json_file,
        sequences,
        # Setup 3: danger-zone based mask, ignore class information
        mode='dz',
        ignore_class=True,
    )
