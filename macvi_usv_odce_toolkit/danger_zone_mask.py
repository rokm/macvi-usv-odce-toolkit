import numpy as np
import cv2


def estimate_plane_from_imu(roll, pitch, height):
    """
    Estimate sea-plane equation from IMU measurements (roll and pitch) and assumed camera height.

    The estimated plane is in USV coordinate system (X: forward, Y: left, Z: up) but with origin
    shifted to the position of the camera's optical center.

    Parameters
    ----------
    roll : float
        Measured roll angle from IMU sensor, in degrees.
    pitch : float
        Measured pitch angle from IMU sensor, in degrees.
    height : float
        Assumed camera height above the sea level/plane.

    Returns
    -------
    plane : numpy.ndarray
        1-D array containing coefficients (A, B, C, D) of plane equation Ax + By + Cz + D = 0.
    """
    # Invert the signs (measured angle -> plane angle)
    pitch = -pitch
    roll = -roll

    # Rotation matrix for roll (rotation around X axis)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])

    # Rotation matrix for pitch (rotation around Y axis)
    c, s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])

    # Normal vector
    n = np.array([[0], [0], [1]])

    # Rotate the normal (@ = matrix multiplication)
    n = Rx @ Ry @ n

    # Plane equation from normal and the height (the constant)
    plane = np.append(n, height)
    plane = plane / np.linalg.norm(plane)

    return plane


def construct_mask_from_danger_zone(
    roll,
    pitch,
    camera_height,
    danger_zone_range,
    camera_matrix,
    dist_coeffs,
    image_width,
    image_height,
    camera_fov=80,
    image_margin=10,
):
    """"
    Construct ignore mask for danger zone, based on IMU measurements and calibrated camera.

    The mask is obtained by projecting sampled points from the danger-zone's edge into the image, in order to obtain a
    polygon that corresponds to the danger zone area in the image. Pixels inside the polygon are marked as area of
    interest, while pixels outside the polygon are marked as ignored.

    Parameters
    ----------
    roll : float
        Measured roll angle from IMU sensor, in degrees.
    pitch : float
        Measured pitch angle from IMU sensor, in degrees.
    camera_height : foat
        Assumed camera height above the sea level/plane.
    danger_zone_range : float
        The radius of the danger zone, in meters.
    camera_matrix : np.ndarray
        Camera (instrinsics) matrix from camera calibration.
    dist_coeffs : np.ndarray
        Camera distortion coefficients from camera calibration.
    image_width : (int, int)
        Image width, in pixels.
    image_height : int
        Image height, in pixels.
    camera_fov : float, optional
        Estimated camera horizontal field of view, in degrees. Used to sample danger-zone edge points for projection
        into the image.
    image_margin : int, optional
        Extra margin value when deciding whether projected point still falls within image boundaries or not.

    Returns
    -------
    mask : numpy.ndarray
        A 2D mask of type numpy.uint8, with size corresponding to image height and image width. Pixels belonging to
        danger zone are set to zero, while pixels outside of the danger zone are set to 255 (= "ignore").
    """
    # Estimate sea plane from IMU pitch and roll and known camera heigh
    A, B, C, D = estimate_plane_from_imu(roll, pitch, camera_height)

    # Sample the points on the border of the danger zone
    num_samples = int(np.ceil(camera_fov)) * 2  # 0.5 degree resolution

    r = np.linspace(90 - (camera_fov / 2), 90 + (camera_fov / 2), num_samples)
    r = np.radians(r)

    x = danger_zone_range * np.sin(r)
    y = danger_zone_range * np.cos(r)
    z = -(A * x + B * y + D) / C

    points = np.transpose(np.array([-y, -z, x]))  # World C.S. to camera C.S.
    projected_points, _ = cv2.projectPoints(
        points,
        np.identity(3),
        np.zeros([1, 3]),
        camera_matrix,
        distCoeffs=dist_coeffs,
    )

    polygon = []
    for projected_point in projected_points:
        # Check if distorted point falls within the image boundaries
        x = projected_point[0, 0]
        y = projected_point[0, 1]

        if not (-image_margin <= x <= (image_width + image_margin)):
            continue
        if not (-image_margin <= y <= (image_height + image_margin)):
            continue

        polygon.append([int(x), int(y)])

    # Close the polygon
    y_first = polygon[0][1]  # y coordinate of first polygon point
    y_last = polygon[-1][1]  # y coordinate of last polygon point
    polygon = (
        [[0, image_height], [0, y_first]] +
        polygon +
        [[image_width, y_last], [image_width, image_height], [0, image_height]]
    )  # yapf: disable

    # Draw the polygon, creating the zeroed-out zone on the sea plane
    mask = 255 * np.ones((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, np.array([polygon]), color=0)

    return mask
