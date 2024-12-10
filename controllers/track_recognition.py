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
