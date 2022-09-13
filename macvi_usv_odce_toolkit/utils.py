import numpy as np


def bbox_in_mask(mask, rect, thr=0.5):
    """
    Check whether the overlap of the given bounding box rectangle with the mask exceeds the specified threshold.

    Parameters
    ----------
    mask : numpy.ndarray
        A 2D mask with 0/1 values.
    rect : iterable
        An iterable containing bounding box rectangle: (x, y, w, h)
    thr : float, optional
        Overlap threshold.

    Returns
    -------
    bool
        A boolean indicating that overlap exceeds the specified threshold.
    """
    x, y, w, h = (int(round(x)) for x in rect)
    roi = mask[y:(y + h), x:(x + w)]
    overlap = np.sum(roi)
    return bool((overlap / (w * h)) > thr)  # np.bool -> bool


def compute_iou_overlaps(rect, annotations, thr=0.3):
    """
    Compute intersection-over-union overlaps between the given bounding-box rectangle and all annotated bounding-box
    rectangles.

    Parameters
    ----------
    rect : iterable
        An iterable containing bounding box rectangle: (x, y, w, h)
    annotations: iterable
        An iterable containing annotated bounding boxes. Each element is a dictionary with a key named 'bbox'
        and corresponding value being an iterable describing the bounding box (x, y, w, h).

    Returns
    -------
    overlaps : iterable
        An iterable containing the overlap value for each annotation. If overlap value is equal or less than
        the specified threshold, it is reset to 0.
    """
    overlaps = [compute_iou(rect, annotation['bbox']) for annotation in annotations]
    return [x if x > thr else 0 for x in overlaps]


def compute_iou(bbox1, bbox2):
    """
    Compute intersection-over-union overlap between two bounding boxes.

    Parameters
    ----------
    bbox1 : iterable
        An iterable containing the first bounding box rectangle: (x, y, w, h).
    bbox2 : iterable
        An iterable containing the second bounding box rectangle: (x, y, w, h).

    Returns
    -------
    iou : float
        IoU overlap between the two bounding boxes.
    """
    bbox1_x1, bbox1_y1, bbox1_w, bbox1_h = bbox1
    bbox2_x1, bbox2_y1, bbox2_w, bbox2_h = bbox2

    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1_x1, bbox2_x1)
    y_top = max(bbox1_y1, bbox2_y1)
    x_right = min(bbox1_x1 + bbox1_w, bbox2_x1 + bbox2_w)
    y_bottom = min(bbox1_y1 + bbox1_h, bbox2_y1 + bbox2_h)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox1_area = bbox1_w * bbox1_h
    bbox2_area = bbox2_w * bbox2_h

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return float(iou)
