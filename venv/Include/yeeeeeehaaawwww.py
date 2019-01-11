import cv2
import time
import numpy as np
from GripPipeline import GripPipeline

# Constants
FOV = 60.0
WIDTH = 640.0
HEIGHT = 480.0
LENGTH_IN_INCHES = 5.5
FOV_PER_PIXEL = 0.09375
X_POSITION = 0
Y_POSITION = 0
ANGLE_OF_CAMERA = 0


# location of text for testing
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# gets camera feed and processes it
cap = cv2.VideoCapture(1)
pipeline = GripPipeline()
while True:
    ret, img=cap.read()
    pipeline.process(img)
    rect = []  # tuple of minAreaRects created from pipeline
    boxes = []  # tuple of a tuple of the corners of the rectangles
    for cont in pipeline.find_contours_output:

        trmp = cv2.minAreaRect(cont)  # creates trmp minAreaRect of the contours from the output
        box = cv2.boxPoints(trmp)  # creates tuple of the corners of trmp
        box = np.int0(box)  # converts all points to integers

        boxes.append(box)
        rect.append(trmp)
    # if there is at least 2 rectangles on the screen, it will draw them and do stuff
    if len(rect) > 1:
        for i in range(1, len(rect)):
            # creates trmp values of 2 consecutive rectangles in rect
            trmp = rect[i]
            trmp2 = rect[i-1]

            # displays the angle of the trmps on the screen
            cv2.putText(img, str(int(trmp[2])), (50, 50), font, 1, (0, 255, 0), 2)
            cv2.putText(img, str(int(trmp2[2])), (50, 100), font, 1, (0, 255, 0), 2)

            # creates a tuple of each of the corners of trmp
            box = cv2.boxPoints(trmp)
            box = np.int0(box)

            # creates a tuple of each of the corners of trmp2
            box2 = cv2.boxPoints(trmp2)
            box2 = np.int0(box2)

            # creates a point that is the center of the first trmp and maps the point to an integer
            p = [(box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4, (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4]
            p2 = list(map(int, p))

            # sorts box and box2
            ySortedBox = list(sorted(box, key=lambda x: x[1]))  # x[1] is the y coordinate
            ySortedBox2 = list(sorted(box2, key=lambda x: x[1]))

            # gets the second highest point in box and box2
            secondHighestPoint = ySortedBox[1][0]
            secondHighestPoint2 = ySortedBox2[1][0]

            # gets the lowest point in box and box2
            lowestPoint = ySortedBox[3][0]
            lowestPoint2 = ySortedBox2[3][0]

            # draws the center points of the first trmp
            img[p2[1], p2[0]] = (0, 0, 255)
            cv2.circle(img, (p2[0], p2[1]), 3, (0, 0, 255))
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # creates a point that is the center of the second trmp and maps the point to an integer
            p3 = [(box2[0][0] + box2[1][0] + box2[2][0] + box2[3][0]) / 4, (box2[0][1] + box2[1][1] + box2[2][1] + box2[3][1]) / 4]
            p4 = list(map(int, p3))

            # draws the center points of the second trmp
            img[p4[1], p4[0]] = (0, 0, 255)
            cv2.circle(img,(p4[0], p4[1]), 3, (0, 0, 255))

            cv2.drawContours(img, [box2], 0, (0, 0, 255), 2)

            if abs(lowestPoint2 - lowestPoint) > abs(secondHighestPoint2 - secondHighestPoint):
                # gets center and draws it
                center = ((p2[0] + p4[0])//2, (p2[1] + p4[1])//2)
                cv2.circle(img, center, 3, (0, 0, 255))

                # gets the distance between the points and the center
                length = abs(ySortedBox[3][0] - center[0])
                print(length)
                angle = FOV_PER_PIXEL * (320 - center[0]) + ANGLE_OF_CAMERA
                cv2.putText(img, str(angle), (50, 150), font, 1, (0, 255, 0), 2)





    # if(len(rect) > 0):

    # cv2.drawContours(img,rect,-1,(0,0,255),2)
    cv2.imshow('first step', img)
    k=cv2.waitKey(10)& 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()