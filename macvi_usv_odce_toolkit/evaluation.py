import os
import json
import tempfile
import contextlib  # redirect_stdout

import cv2
import numpy as np

import pycocotools.coco
import pycocotools.cocoeval

from .dataset import load_camera_calibration
from .danger_zone_mask import construct_mask_from_danger_zone
from .sea_edge_mask import construct_mask_from_sea_edge
from . import utils

# Danger zone parameters
# NOTE: estimated camera HFoV is actually around 66 degrees, but we use 80 degrees to ensure that we project at least
# one sampled point beyond the image borders.
DANGER_ZONE_RANGE = 15  # danger zone range, in meters
DANGER_ZONE_CAMERA_HEIGHT = 1.0  # height of the camera (in meters)
DANGER_ZONE_CAMERA_FOV = 80  # camera HFoV, in degrees
DANGER_ZONE_IMAGE_MARGIN = 10  # image margin, in pixels

# Obstacle classes
OBSTACLE_CLASSES = ('ship', 'person', 'other')
OBSTACLE_CLASS_NAME_TO_ID_MAP = {name: idx for idx, name in enumerate(OBSTACLE_CLASSES)}


def convert_to_coco_structures(dataset_json_file, results_json_file, sequences=None, mode='full', ignore_class=False):
    """
    Convert the dataset annotations and detection results in COCO-compatible data structures.

    Parameters
    ----------
    dataset_json_file : str
        Full path to dataset JSON file.
    results_json_file : str
        Full path to detection results JSON file.
    sequences : iterable, optional
        Optional list of sequence IDs to process. By default, all sequences are processed.
    mode : str, optional
        Evaluation mode:
            'full': use only static ignore mask provided by camera calibration.
            'edge': use sea-edge based ignore mask in addition to static mask.
            'dz': use danger-zone based ignore mask in addition to static mask.
    ignore_class : bool, optional
        Flag indicating whether to ignore class labels or not.

    Returns
    -------
    coco_dataset : dict
        Dictionary containing dataset annotations in COCO-compatible data structure.
    coco_results : list
        List containing detection results in COCO-compatible data structure.
    """

    assert mode in {'edge', 'dz', 'full'}

    sequences = set(sequences) if sequences is not None else set()

    # Load dataset JSON file
    with open(dataset_json_file, 'r') as fp:
        dataset = json.load(fp)
    dataset = dataset['dataset']
    dataset_path = os.path.dirname(dataset_json_file)  # Dataset root directory

    # Load results (detections) file
    with open(results_json_file, 'r') as fp:
        results = json.load(fp)
    results = results['dataset']

    # Select sequences
    if sequences:
        dataset_sequences = [seq for seq in dataset['sequences'] if seq['id'] in sequences]
        results_sequences = [seq for seq in results['sequences'] if seq['id'] in sequences]
    else:
        dataset_sequences = dataset['sequences']
        results_sequences = results['sequences']

    # Sanity check
    assert len(dataset_sequences) == len(results_sequences), "Mismatch in dataset and result sequences length!"

    # Ensure sequences are ordered by ID, just in case
    dataset_sequences.sort(key=lambda seq: seq['id'])
    results_sequences.sort(key=lambda seq: seq['id'])

    # Global lists of images, annoations, and detections - we are going to merge individual sequences into a single one.
    image_entries = []
    annotation_entries = []
    detection_entries = []

    image_id = 0
    annotation_id = 0

    # Iterate over all sequences
    for dataset_sequence, results_sequence in zip(dataset_sequences, results_sequences):
        assert dataset_sequence['id'] == results_sequence['id'], "Dataset and results sequence ID mismatch!"

        # Infer the sequence base name from the frames path. Required for construction of the camera calibration
        # filename. Ideally, both would be part of the metadata, but here we are...
        # The format is fixed, e.g., "/kope100-00006790-00007090/frames/", and we want to extract the "kope100" part.
        sequence_name = dataset_sequence['path'].split('/')[1]  # kope100-00006790-00007090
        sequence_base_name = sequence_name.split('-')[0]  # kope100

        # Sequence-wide exhaustive annotation flag
        sequence_exhaustive = dataset_sequence['exhaustive'] > 0

        # Load camera calibration for the sequence
        calibration_file = os.path.join(dataset_path, 'calibration', f'calibration-{sequence_base_name}.yaml')
        assert os.path.isfile(calibration_file), f"Camera calibration file {calibration_file:!r} does not exist!"
        camera_calibration = load_camera_calibration(calibration_file)

        image_width, image_height = camera_calibration['imageSize']
        assert image_width == 1278
        assert image_height == 958

        # Load mask, if available
        sequence_mask = None
        if 'mask' in dataset_sequence:
            # Alas, the 'mask' attribute does not store correct mask name, which is always 'ignore_mask.png'...
            mask_filename = os.path.join(dataset_path, 'sequences', sequence_name, 'ignore_mask.png')
            sequence_mask = cv2.imread(mask_filename, 0)

        # Iterate over all frames in the sequence
        dataset_frames = dataset_sequence['frames']
        results_frames = results_sequence['frames']
        for dataset_frame, results_frame in zip(dataset_frames, results_frames):
            assert dataset_frame['id'] == results_frame['id'], "Dataset and results frame ID mismatch!"

            # Per-frame exhaustive annotation flag. If sequence_exhaustive flag is False, so is frame_exhaustive.
            # Otherwise, frame's metadata can override the sequence-wide flag on per-frame setting.
            if sequence_exhaustive:
                frame_exhaustive = sequence_exhaustive
                if 'exhaustive' in dataset_frame:
                    frame_exhaustive = dataset_frame['exhaustive'] > 0
            else:
                frame_exhaustive = False  # Disabled on sequence level.

            # Construct mask
            # If frame is exhaustively annotated, construct mode-specific
            # mask, and combine it with the static mask. If not, construct
            # mask only in danger-zone mode (to evaluate detections in
            # evaluate mode); in other modes, ignore all detections in
            # that frame.
            if frame_exhaustive:
                mask = np.zeros((image_height, image_width), dtype=np.uint8)

                if mode == 'edge':
                    # Sea-edge based mask
                    mask |= construct_mask_from_sea_edge(
                        dataset_frame['water_edges'],
                        image_width,
                        image_height,
                    )
                elif mode == 'dz':
                    # Danger-zone based mask
                    mask |= construct_mask_from_danger_zone(
                        roll=dataset_frame['roll'],
                        pitch=dataset_frame['pitch'],
                        camera_height=DANGER_ZONE_CAMERA_HEIGHT,
                        danger_zone_range=DANGER_ZONE_RANGE,
                        camera_matrix=camera_calibration['M1'],
                        dist_coeffs=camera_calibration['D1'],
                        image_width=image_width,
                        image_height=image_height,
                        camera_fov=DANGER_ZONE_CAMERA_FOV,
                        image_margin=DANGER_ZONE_IMAGE_MARGIN,
                    )

                # Apply sequence-wide static mask, if available
                if sequence_mask is not None:
                    mask |= sequence_mask

                mask[mask > 0] = 1  # Turn into 0/1 valued mask
            else:
                if mode == 'dz':
                    # Danger-zone based mask
                    mask = construct_mask_from_danger_zone(
                        roll=dataset_frame['roll'],
                        pitch=dataset_frame['pitch'],
                        camera_height=DANGER_ZONE_CAMERA_HEIGHT,
                        danger_zone_range=DANGER_ZONE_RANGE,
                        camera_matrix=camera_calibration['M1'],
                        dist_coeffs=camera_calibration['D1'],
                        image_width=image_width,
                        image_height=image_height,
                        camera_fov=DANGER_ZONE_CAMERA_FOV,
                        image_margin=DANGER_ZONE_IMAGE_MARGIN,
                    )

                    # Apply sequence-wide static mask, if available
                    if sequence_mask:
                        mask |= sequence_mask

                    mask[mask > 0] = 1  # Turn into 0/1 valued mask
                else:
                    # Ignore all detections in the frame.
                    mask = np.ones((image_height, image_width), dtype=np.uint8)

            # Process annotated and detected obstacles
            annotated_obstacles = dataset_frame.get('obstacles', [])
            detected_obstacles = results_frame.get('detections', [])

            for annotated_obstacle in annotated_obstacles:
                bbox = annotated_obstacle['bbox']
                # Add negative annotations to the mask
                if annotated_obstacle['type'] == 'negative':
                    x, y, w, h = bbox
                    mask[y:(y + h), x:(x + w)] = 1
                else:
                    # Check whether the annotation lies in the ignore region and if it overlaps any detected obstacle.
                    ignore = utils.bbox_in_mask(mask, bbox)
                    overlap_values = utils.compute_iou_overlaps(bbox, detected_obstacles)

                    if ignore and not any(overlap_values):
                        if mode == 'dz':
                            continue

                    if ignore_class:
                        class_id = 0
                    else:
                        class_id = OBSTACLE_CLASS_NAME_TO_ID_MAP[annotated_obstacle['type']]

                    annotation_entries.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': bbox,
                        'iscrowd': 0,
                        'area': annotated_obstacle['area'],
                        'segmentation': [],
                        'ignore': int(ignore),  # bool -> int
                    })
                    annotation_id += 1  # Increment global annotation ID

            for detected_obstacle in detected_obstacles:
                bbox = detected_obstacle['bbox']
                ignore = utils.bbox_in_mask(mask, bbox)
                overlap_values = utils.compute_iou_overlaps(bbox, annotated_obstacles)
                if ignore and not any(overlap_values):
                    if mode == 'dz':
                        continue

                if ignore_class:
                    class_id = 0
                else:
                    # Allow detection type to be either name or numeric class ID
                    class_id = detected_obstacle['type']
                    if isinstance(class_id, str):
                        class_id = OBSTACLE_CLASS_NAME_TO_ID_MAP[class_id]

                detection_entries.append({
                    'image_id': image_id,
                    'category_id': class_id,
                    'bbox': bbox,
                    'score': 1,
                    'ignore': int(ignore),  # bool -> int
                })

            image_entries.append({
                'id': image_id,
                'width': image_width,
                'height': image_height,
                'file_name': dataset_frame['image_file_name'],
            })
            image_id += 1  # Increment global image ID

    # COCO dataset/ground truth structure
    coco_dataset = {
        'info': {
            'year': 2022,
        },
        'categories': [{
            'id': idx,
            'name': name,
            'supercategory': 'obstacle',
        } for idx, name in enumerate(OBSTACLE_CLASSES)],
        'annotations': annotation_entries,
        'images': image_entries,
    }

    # COCO results
    coco_results = detection_entries

    return coco_dataset, coco_results


def evaluate_detection_results(dataset_json_file, results_json_file, sequences=None, mode='full', ignore_class=False):
    """
    Evaluate detection results.

    This function loads the dataset annotations and detection results from their respective files, converts them to
    COCO-compatible data structures, and performs evaluation using pycocotools.

    Parameters
    ----------
    dataset_json_file : str
        Full path to dataset JSON file.
    results_json_file : str
        Full path to detection results JSON file.
    sequences : iterable, optional
        Optional list of sequence IDs to process. By default, all sequences are processed.
    mode : str, optional
        Evaluation mode:
            'full': use only static ignore mask provided by camera calibration.
            'edge': use sea-edge based ignore mask in addition to static mask.
            'dz': use danger-zone based ignore mask in addition to static mask.
    ignore_class : bool, optional
        Flag indicating whether to ignore class labels or not.

    Returns
    -------
    f_scores : tuple
        A four-element tuple containing F-score values: F_all, F_small, F_medium, and F_large.
    """

    # Convert annotations and results to COCO-compatible structures
    dataset_dict, results_list = convert_to_coco_structures(
        dataset_json_file,
        results_json_file,
        sequences,
        mode,
        ignore_class,
    )

    # Capture pycocotools' output to prevent spamming stdout with its diagnostic messages
    with contextlib.redirect_stdout(None):
        # Initialize COCO helper classes from in-memory data, to avoid having to write them to temporary files...
        coco_dataset = pycocotools.coco.COCO()
        # This is equivalent to passing filename to pycocotools.coco.COCO()
        coco_dataset.dataset = dataset_dict
        coco_dataset.createIndex()

        # coco_dataset.loadRes() can be passed either filename or a list
        coco_results = coco_dataset.loadRes(results_list)

        # Create evaluation...
        coco_evaluation = pycocotools.cocoeval.COCOeval(coco_dataset, coco_results, iouType='bbox')
        coco_evaluation.params.iouThrs = np.array([0.3, 0.3])  # IoU thresholds for evaluation

        # ... and evaluate
        coco_evaluation.evaluate()
        coco_evaluation.accumulate()
        coco_evaluation.summarize()

    stats = np.nan_to_num(coco_evaluation.stats)
    stats[stats == -1] = 0

    # Compute F-scores
    def _f_score(precision, recall):
        if precision != 0 and recall != 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0

    f_all = _f_score(stats[0], stats[8])
    f_small = _f_score(stats[3], stats[9])
    f_medium = _f_score(stats[4], stats[10])
    f_large = _f_score(stats[5], stats[11])

    return f_all, f_small, f_medium, f_large
