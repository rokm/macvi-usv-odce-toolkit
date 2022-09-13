import cv2


def load_camera_calibration(filename):
    """
    Load camera calibration from the specified YAML file.

    Parameters
    ----------
    filename : str
        Name of the calibration YAML file to load.

    Returns
    -------
    calibration : dict
        A dictionary containing the camera calibration data (M1, M2, D1, D2, R, T, imageSize).
    """
    storage = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    # These fields are serialized as matrices...
    _MAT_KEYS = 'M1', 'M2', 'D1', 'D2', 'R', 'T'
    calibration = {key: storage.getNode(key).mat() for key in _MAT_KEYS}

    # imageSize is serialized as a sequence, so it needs special handling.
    node = storage.getNode('imageSize')
    assert node.isSeq() and node.size() == 2

    node_width = node.at(0)
    node_height = node.at(1)

    assert node_width.isReal() or node_width.isInt()
    assert node_height.isReal() or node_height.isInt()

    calibration['imageSize'] = int(node_width.real()), int(node_height.real())

    return calibration
