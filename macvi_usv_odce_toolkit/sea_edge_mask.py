import cv2
import numpy as np


def construct_mask_from_sea_edge(sea_edges, image_width, image_height):
    """"
    Construct ignore mask based on the annotated sea edge.

    The mask is obtained by constructing polygon from the give sea-edge annotations, and filling the polygon below the
    edge. The pixels in the area below the sea edge are marked as area of interest, while pixels in the area above it
    are marked as ignored.

    Parameters
    ----------
    sea_edges : iterable
        An iterable containing one or more sea-edge annotations. Each annotation is a dictionary with two elements,
        x_axis and y_axis, each being an interable containing x and y image coordinates, respectively.
    image_width : (int, int)
        Image width, in pixels.
    image_height : int
        Image height, in pixels.

    Returns
    -------
    mask : numpy.ndarray
        A 2D mask of type numpy.uint8, with size corresponding to image height and image width. Pixels in the area below
        the sea edge are set to zero, while pixels in the are above the sea edge are set to 255 (= "ignore").
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    polygons = []
    for sea_edge in sea_edges:
        x_values = sea_edge['x_axis']
        y_values = sea_edge['y_axis']

        if not x_values or not y_values:
            continue

        x_values = [x_values[0]] + x_values + [x_values[-1]]
        y_values = [0] + y_values + [0]

        polygons.append(np.array([[int(x), int(y)] for x, y in zip(x_values, y_values)]))

    cv2.fillPoly(mask, pts=polygons, color=(255, 255, 255))
    return mask
