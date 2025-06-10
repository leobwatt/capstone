import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

BUFFER_SIZE = 64
TOLERANCE = 10

points: deque[tuple[int, int]] = deque(maxlen=BUFFER_SIZE)

red_lower = np.array((147, 141, 146))
red_upper = np.array((182, 175, 255))


def process_points(points: deque[tuple[int, int]]) -> tuple[float, float]:
    """
    Process the position points and return a tuple with the velocity and angle.
    """

    if len(points) < 2:
        return 0.0, 0.0

    # Calculate velocity
    try:
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
    except IndexError:
        return 0.0, 0.0

    velocity = (dx**2 + dy**2) ** 0.5

    # Calculate angle
    angle = np.arctan2(dy, dx) * 180 / np.pi

    return velocity, angle


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/json")
        self.end_headers()

        velocity, angle = process_points(points)
        response = {"velocity": velocity, "angle": angle}
        self.wfile.write(json.dumps(response).encode("utf-8"))


def run_server():
    server = HTTPServer(("localhost", 8080), RequestHandler)
    print("Starting server on http://localhost:8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")
        server.server_close()


server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

capture = cv2.VideoCapture(0)
# missed_frames = 0

while True:
    _, frame = capture.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, red_lower, red_upper)
    mask = cv2.erode(mask, np.zeros(1), iterations=2)
    mask = cv2.dilate(mask, np.zeros(1), iterations=2)

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    center = (0, 0)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        m = cv2.moments(c)
        try:
            center = (int(m["m10"] / (m["m00"])), int(m["m01"] / (m["m00"] + 0.01)))
        except ZeroDivisionError:
            # missed_frames += 1
            # if missed_frames > TOLERANCE:
            #     print("Missed too many frames, exiting.")
            #     break
            continue

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    points.appendleft(center)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1) * 2.5))
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
