import cv2
import numpy
import math
from enum import Enum

class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__hsv_threshold_hue = [36.6546762589928, 102.07130730050935]
        self.__hsv_threshold_saturation = [36.6906474820144, 255.0]
        self.__hsv_threshold_value = [82.55395683453237, 255.0]

        self.hsv_threshold_output = None


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step HSV_Threshold0:
        self.__hsv_threshold_input = source0
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)


    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

import os
import numpy as np
import sys

cam_mtrx =  np.array(
[[859.68420929,   0.,         318.24497556],
 [  0.,         891.25688996, 225.79027452],
 [  0.,           0.,           1.        ]], dtype='float32')

cam_mtrx =  np.array(
[[715,   0.,         310],
 [  0.,         715, 240],
 [  0.,           0.,           1.        ]], dtype='float32')

distorts = None#np.array([-0.07893495,  1.34223772, -0.00841574, -0.008392,   -4.1361857 ], dtype='float32')
world_pts = [
        (3.313, 4.824),
        (1.337, 5.325),
        (0, 0),
        (1.936, -0.501),
        (13.290, 5.325),
        (11.314, 4.824),
        (12.691, -0.501),
        (14.627, 0),
]

def draw(img, corner, imgpts):
    corner = tuple(corner.ravel())
    try:
        print(tuple(imgpts[0].ravel()))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    except OverflowError:
        pass
    return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

def draw_cube(img, corner, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img
cube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

DEV = 2
#world_pts += ((14.627 - x, y) for x, y in reversed(world_pts))
world_pts = np.float32([(x, -y, 0) for x, y in world_pts])
print(world_pts)
print(cam_mtrx)
cap = cv2.VideoCapture(DEV)
print('Got cap')
exposure = 80 if len(sys.argv) < 2 else int(sys.argv[1])
os.system('v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c white_balance_temperature_auto=0 -c exposure_absolute={}'.format(DEV, exposure))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap.set(cv2.CAP_PROP_EXPOSURE, n)
pipeline = GripPipeline()
targetColor = cv2.cvtColor(np.array([[[128, 255, 128]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
lab_factors = 2, 1, 1
import itertools
while True:
    rv, fr = cap.read()
    fr_lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    channels = cv2.split(fr_lab.astype('int16'))
    greenscale = np.zeros(fr.shape[:-1], 'int16')
    for tr, ch, fac in zip(targetColor, channels, lab_factors):
        greenscale += np.absolute(ch - tr) // fac
    #print(greenscale.max())
    greenscale = 255 - np.clip((greenscale), 0, 255).astype('uint8')
    cv2.imshow('greenscale', greenscale)
    if not rv: break
    pipeline.process(fr)
    op = pipeline.hsv_threshold_output
    o8 = op.astype('uint8') * 128
    corners = cv2.cornerHarris(np.float32(greenscale), 2, 3, 0.03)
    #print(corners.min(), corners.max())
    #cm = corners + (-corners.min() + 1)
    #print(cm.min(), cm.max())
    #lcorn = np.log(cm)
    
    #print(lcorn.min(), lcorn.max())
    #cv2.imshow('corners', lcorn / 16)
    kernel = np.ones((5, 5), np.uint8)
    eroded_hsv = cv2.dilate(cv2.erode(op, kernel, iterations=1), kernel, iterations=3)#.astype(np.bool_)
    contours, hier = cv2.findContours(eroded_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_c = itertools.islice(sorted(contours, key=cv2.contourArea, reverse=True), 2)
    zer = np.zeros(eroded_hsv.shape, np.uint8)
    for i in biggest_c:
        cv2.fillPoly(zer, pts=[i], color=255)
        cv2.drawContours(fr, i, -1, (255, 255, 255), 1)
    cactual = (corners > 0.007 * corners.max()) & zer.astype(np.bool_) #eroded_hsv

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(cactual.astype('uint8'))
    fr[cactual] = (0, 0, 255)
    fc = []
    for x, y in centroids[1:]:
            #fr[int(y), int(x)] = (0, 255, 0)
            ix, iy = int(x), int(y)
            if 30 < op[iy-10:iy+10,ix-10:ix+10].astype(np.bool_).sum() < 180:
                #cv2.circle(fr, (int(x), int(y)), 2, (0, 255, 0), -1)
                fc.append((x, y))
            else: pass
                #cv2.circle(fr, (int(x), int(y)), 2, (0, 0, 255), -1)
    c_exact = []
    if fc:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        c_exact = cv2.cornerSubPix(greenscale, np.float32(fc), (5, 5), (-1, -1), criteria)
        #for x, y in c_exact:
        #    cv2.circle(fr, (int(x), int(y)), 1, (255, 0, 0), -1)
        #for i, (x, y, z) in enumerate(world_pts):
        #    cv2.putText(fr,str(i),(int(x*5),int(y*5)+50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
        
        if len(c_exact) == 8:
            h = list(sorted(c_exact, key=lambda x: x[0]))
            left = h[:4]
            right = h[4:]
            pts = []
            for rect in left, right:
                centerx, centery = sum(x[0] for x in rect) / len(rect), sum(x[1] for x in rect) / len(rect)
                def key(r):
                    x, y = r
                    dx, dy = x - centerx, y - centery
                    return math.atan2(-dy, dx) % (2 * math.pi)
                pts += sorted(rect, key=key)
            for i, (x, y) in enumerate(pts):
                cv2.putText(fr,str(i),(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
            pts = np.float32(pts)
            #print(pts)
            retval2, rvecs, tvecs, inliers = cv2.solvePnPRansac(world_pts, pts, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
            cv2.putText(fr,'I: {}'.format(len(inliers) if inliers is not None else 'X'),(550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255) if inliers is not None and len(inliers) == 8 else (0,0,255),1,cv2.LINE_AA)
            #retval2, rvecs, tvecs = cv2.solvePnP(world_pts, pts, cam_mtrx, distorts)
            #print(inliers)
            #print(rvecs, tvecs)
            if inliers is not None:
                centerp = np.float32([14.627/2, -5.325/2, 0])
                #ax2 = axis + centerp
                #print(ax2)
                #ax2 = np.insert(ax2, 0, centerp, axis=0)#np.float32([centerp]) + ax2
                #print(ax2)
                #imgpts, jac = cv2.projectPoints(ax2, rvecs, tvecs, cam_mtrx, distorts)
                #fr =draw(fr, imgpts[0], imgpts[1:])

                qb = cube + centerp
                imgpts, jac = cv2.projectPoints(qb, rvecs, tvecs, cam_mtrx, distorts)
                fr = draw_cube(fr, None, imgpts)

                pts2, jac = cv2.projectPoints(world_pts, rvecs, tvecs, cam_mtrx, distorts)
                for x, y in np.int32(pts2).reshape(-1, 2):
                    cv2.circle(fr, (x, y), 2, (255, 255, 255), -1)
        else:
            cv2.putText(fr,'C: {}'.format(len(c_exact)),(550, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)



    #pipeline.hsv_threshold_output
    #print(corners)
    cv2.imshow('dsst', op)
    cv2.imshow('bc', zer)
    cv2.imshow('f1', fr)
    if cv2.waitKey(1) & 0xff == 27: break
