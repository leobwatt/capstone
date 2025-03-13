from collections import deque

import cv2
import numpy as np

BUFFER_SIZE = 64


red_lower = np.array((147, 141, 146))
red_upper = np.array((182, 175, 255))
points: deque[tuple[int, int]] = deque(maxlen=BUFFER_SIZE)

capture = cv2.VideoCapture(0)
prev_center = None

while True:
    _, frame = capture.read()

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, red_lower, red_upper)
    mask = cv2.erode(mask, np.zeros(1), iterations=2)
    mask = cv2.dilate(mask, np.zeros(1), iterations=2)
    cv2.imwrite("test.png", mask)

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    center = (0, 0)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        m = cv2.moments(c)
        center = (int(m["m10"] / (m["m00"] + 0.01)), int(m["m01"] / (m["m00"] + 0.01)))

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
