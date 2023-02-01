import cv2
from util import ARUCO_TAGS
from math import atan

# Start a video stream
stream = cv2.VideoCapture(0)

def detect_markers():
    """Detects and returns data about ArUco markers in the current frame."""

    # Get the current frame
    frame = stream.read()

    # Check if the frame was grabbed
    if not frame:
        return

    # Convert the frame to grayscale and detect ArUco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, ARUCO_TAGS)

    # Check if ArUco markers were detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()

        tag_positions = []

        # Loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # Extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            tag_positions.append((topLeft, topRight, bottomRight, bottomLeft))

    return tag_positions

def skew_to_euler(skew_x, skew_y):
    """Converts skew to Euler angles."""

    # Calculate the Euler angles
    # Involves arctangent, but I do not yet understand

    return (x, y, z)

def get_tag_data(tag_positions):
    """Returns the skew and distance of a tag."""
    (topLeft, topRight, bottomRight, bottomLeft) = tag_positions

    midpoint = ((topLeft[0] + topRight[0]) // 2, (topLeft[1] + bottomLeft[1]) // 2)
    skew = (topLeft[0] - topRight[0]) / (topLeft[1] - topRight[1])
    distance = ((topLeft[0] - topRight[0]) ** 2 + (topLeft[1] - topRight[1]) ** 2) ** 0.5

    return (midpoint, skew, distance)

# Release the VideoCapture object
stream.release()