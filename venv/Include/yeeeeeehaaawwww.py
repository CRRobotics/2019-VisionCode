import cv2
import time
import numpy as np
from GripPipeline import GripPipeline

FOV = 60.0
WIDTH = 640.0
HEIGHT = 480.0
FOV_PER_PIXEL = FOV/WIDTH

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cap = cv2.VideoCapture(1)
#cap.set(15, 1)
pipeline = GripPipeline()
while True:
    ret,img=cap.read()
    #cv2.imshow('video capture',img)
    pipeline.process(img)
    rect = []
    boxes = []
    for cont in pipeline.find_contours_output:

        trmp = cv2.minAreaRect(cont)
        cv2.putText(img, str(trmp[2]), (50, 50), font, 1, (0, 255, 0), 2)
        #print(trmp[2])
        box = cv2.boxPoints(trmp)
        box = np.int0(box)
        boxes.append(box)
        p = [(box[0][0]+box[1][0]+box[2][0]+box[3][0])/4,(box[0][1]+box[1][1]+box[2][1]+box[3][1])/4]
        #cv2.drawMarker(img,p,(0,0,255))
        #print(p)
        p2 = list(map(int, p))
        img[p2[1], p2[0]] = (0,0,255)
        cv2.circle(img,(p2[0],p2[1]),3,(0,0,255))
        # print(box);
        print()
        #print(len(cont))
        cv2.drawContours(img,[box],0,(0,0,255),2)
        rect.append(trmp)
    #if(len(rect) > 0):

    #cv2.drawContours(img,rect,-1,(0,0,255),2)
    cv2.imshow('first step', img)
    k=cv2.waitKey(10)& 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()


#(0..3).map {  }