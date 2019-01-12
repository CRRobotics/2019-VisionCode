import cv2
import numpy as np
from grip import GripPipeline
import argparse

cap = cv2.VideoCapture(1)
pipeline = GripPipeline()
while True:
    # Capture frame-by-frame
    ret, frame_in = cap.read()
    pipeline.process(frame_in)
    # print(frame_in)
    frame = np.zeros(frame_in.shape[:-1], frame_in.dtype)
    # print(frame)
    # cv2.drawContours(frame, pipeline.find_contours_output, -1, 255, 2)
    cv2.fillPoly(frame, pts=pipeline.find_contours_output, color=255)

    # frame = frame_in
    # Our operations on the frame come here
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.6, 100)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (200, 15, 15), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame_in', frame_in)
    cv2.imshow('frame', frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
