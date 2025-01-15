import numpy as np
import cv2


def extract_track(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # resize image
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # convert to 8-bit if it's not 8-bit yet
    if 0 <= np.min(img) <= 1 and 0 <= np.max(img) <= 1:
        img *= 255
    img = img.astype(np.uint8)
    # binary threshold
    img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(img)

    # connected components
    _, object_labels = cv2.connectedComponents(img)

    # remove non-bottom row components
    bottom_row = object_labels[-1]
    bottom_labels = np.unique(bottom_row)
    object_labels = np.where(np.isin(object_labels, bottom_labels), object_labels, 0)
    object_labels = np.where(object_labels > 0, 1, 0).astype(np.uint8)

    return object_labels


def process_track_into_line(processed_image):
    rotated_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
    if rotated_image is None:
        return 0, 0
    # linear regression
    y, x = np.where(rotated_image > 0)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c


def get_track_properties(m, c, track_image_shape):
    if m == 0 and c == 0:
        return 0, 0
    track_direction = m
    track_offset_from_the_middle = (c / track_image_shape[1] - 0.5) * 2
    return track_direction, track_offset_from_the_middle


def get_track_line_coordinates(
    track_direction, track_offset_from_the_middle, track_image_shape
):
    track_start_coordinates = (
        track_image_shape[0] - 1,
        (track_offset_from_the_middle + 1) / 2 * track_image_shape[1],
    )
    track_end_coordinates = (
        track_image_shape[0] // 2,
        track_start_coordinates[1] + track_direction * track_image_shape[0] // 2,
    )
    return track_start_coordinates, track_end_coordinates


def simulate_camera_view(track_image, camera_position, z_angle, side_length):
    x, y = camera_position
    half_side = side_length / 2

    offset_x = -half_side * np.sin(np.radians(z_angle))
    offset_y = -half_side * np.cos(np.radians(z_angle))

    center_x = x + offset_x
    center_y = y + offset_y

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), 360 - z_angle, 1.0)

    height, width = track_image.shape[:2]
    rotated_image = cv2.warpAffine(track_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    start_x = int(center_x - half_side)
    start_y = int(center_y - half_side)
    end_x = int(center_x + half_side)
    end_y = int(center_y + half_side)

    cropped_square = rotated_image[start_y:end_y, start_x:end_x]

    return cropped_square