import cv2
from util import ARUCO_TAGS
from math import atan

# Start a video stream
stream = cv2.VideoCapture(0)

def detect_markers(frame):
    """Detects and returns data about ArUco markers in the current frame."""

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
    raise NotImplementedError

def get_tag_data(tag_positions):
    """Returns the skew and distance of a tag."""
    (topLeft, topRight, bottomRight, bottomLeft) = tag_positions

    midpoint = ((topLeft[0] + topRight[0]) // 2, (topLeft[1] + bottomLeft[1]) // 2)
    distance = ((topLeft[0] - topRight[0]) ** 2 + (topLeft[1] - topRight[1]) ** 2) ** 0.5

    skew_x = atan((topLeft[1] - topRight[1]) / (topLeft[0] - topRight[0]))
    skew_y = atan((topLeft[1] - bottomLeft[1]) / (topLeft[0] - bottomLeft[0]))

    return (midpoint, (skew_x, skew_y), distance)

def main():
    """Main function."""

    while True:

        # Get the current frame
        _, frame = stream.read()

        if not frame:
            print('no fram :(')
            continue

        if DEBUG_GUI:
            debug.update(frame)
        
        # Detect markers
        #tag_positions = detect_markers(frame)



        # Check if markers were detected
        #if tag_positions:

            # Get the data of the first tag
        #    midpoint, skew, distance = get_tag_data(tag_positions[0])

        #    print(skew)

if __name__ == "__main__":
    main()

# Release the VideoCapture object
stream.release()