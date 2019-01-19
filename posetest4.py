import cv2
import numpy
import math
from enum import Enum
import sys

constants = {}
s = exec(open(sys.argv[1], 'r').read(), constants)


class GripPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__hsv_threshold_hue = constants['hsv_threshold_hue']
        self.__hsv_threshold_saturation = constants['hsv_threshold_saturation']
        self.__hsv_threshold_value = constants['hsv_threshold_value']


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

cam_mtrx = constants['cam_mtrx']
distorts = constants['distorts']

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
        #print(tuple(imgpts[0].ravel()))
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

DEV = int(sys.argv[2])
world_pts = np.float32([(x, -y, 0) for x, y in world_pts])
print(world_pts)
print(cam_mtrx)
cap = cv2.VideoCapture(DEV)
print('Got cap')
exposure = 80 if len(sys.argv) < 4 else int(sys.argv[3])
os.system('v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c white_balance_temperature_auto=0 -c exposure_absolute={}'.format(DEV, exposure))
os.system('v4l2-ctl -d /dev/video{} -c focus_auto=0'.format(DEV))
os.system('v4l2-ctl -d /dev/video{} -c focus_absolute=0'.format(DEV))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
pipeline = GripPipeline()
targetColor = constants['targetColor']
lab_factors = 2, 1, 1
g_rot = np.float32([[0, 0, 0]]).T
g_pos = np.float32([[0, 0, 0]]).T
import itertools
import scipy.signal 
def main_loop():
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
        o8 = op.astype('uint8') * 255
        corners = cv2.cornerHarris(np.float32(greenscale), 2, 3, 0.03)
        #corners[corners < 0] = 0
        #print(corners.min(), corners.max())
        #cm = corners + (-corners.min() + 1)
        #print(cm.min(), cm.max())
        cm = corners.copy()
        cm[cm < 0] = 0
        #print(cm.max())
        cm = cv2.cvtColor((cm / 65536).astype(np.float32),cv2.COLOR_GRAY2RGB)
        #lcorn = np.log(cm)
        
        #print(lcorn.min(), lcorn.max())
        #cv2.imshow('corners', lcorn / 16)
        kernel = np.ones((5, 5), np.uint8)
        eroded_hsv_1 = cv2.dilate(cv2.erode(op, kernel, iterations=1), kernel, iterations=1)#.astype(np.bool_)
        contours1, hier1 = cv2.findContours(eroded_hsv_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approxes = [cv2.approxPolyDP(x, 0.01 * cv2.arcLength(x, True), True) for x in contours1]
        #for i in approxes:
        #    print(i)
        #    cv2.drawContours(fr, i, -1, (255, 255, 0), 3)
        #print('approxes', approxes)
        a = False
        pts1 = []#np.zeros((0, 2), dtype=np.float32)
        prevArea = 0
        for area, loop in sorted(((cv2.contourArea(x), x) for x in approxes), key=lambda x: x[0], reverse=True):
            if len(loop) > 8: continue
            if area < prevArea / 2: continue
            prevArea = area
            #x = list(sorted(enumerate(loop), key=lambda x: x[1][0][1]))
            #median_idx, last = x[len(x) // 2]
            #l2 = np.concatenate((loop[median_idx:], loop[:median_idx]), axis=0)


            #print('loop', loop)
            angles = np.zeros(len(loop), dtype='float32')
            #dsts = np.zeros(len(loop), dtype='float32')
            ll = len(loop)
            for i, cur in enumerate(loop):
                prev = loop[(i-1)%ll][0]
                nex = loop[(i+1)%ll][0]
                cur = cur[0]
                angles[i] = (math.atan2(cur[1] - prev[1], cur[0] - prev[0]) - math.atan2(cur[1] - nex[1], cur[0] - nex[0])) % math.pi
            sp = [x[0] for i, x in enumerate(loop) if angles[i] < math.pi * .85]
            if len(sp) < 4: continue
            #sp = list(sorted(enumerate(loop), key=lambda x: angles[x[0]], reverse=True))

            #if len(sp) > 4 and angles[sp[4][0]] > 0.1 * angles[sp[3][0]]:
            #    cv2.polylines(fr, [loop], True, (0, 0, 255))
            #    continue
            pairs = [(sp[i], sp[(i+1)%len(sp)]) for i in range(len(sp))]
            relevantLines = list(sorted(pairs, key=lambda x: ((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2), reverse=True))[:4]
            order = relevantLines[0], relevantLines[2], relevantLines[1], relevantLines[3]
            for i, cur in enumerate(order):
                nex = order[(i+1)%4]
                (x1, y1), (x2, y2) = cur
                (x3, y3), (x4, y4) = nex
                try:
                    res = np.linalg.solve(np.array([
                        [x2 - x1, 0, -1, 0],
                        [y2 - y1, 0, 0, -1],
                        [0, x4 - x3, -1, 0],
                        [0, y4 - y3, 0, -1]]), np.array([[-x1, -y1, -x3, -y3]]).T)
                except np.linalg.LinAlgError:
                    continue
                #print(res)
                #cv2.circle(fr, (int(res[2]), int(res[3])), 2, (255, 0, 255), -1)
                pts1.append(res[2:4,0])
            #angles2 = [math.atan2(x[1][1] - x[0][1], x[0][1] - x[0][0]) for x in relevantLines]
            #sg = segment_by_angle_kmeans(list(range(4)), angles2)
            #print(sg)

            #for l1, l2 in relevantLines:
            #    cv2.line(fr, tuple(l1), tuple(l2), (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.polylines(fr, [loop], True, (0, 255, 0))
            sp = [x[1] for x in sp[:4]]
            #print(sp)
            #fc += np.float32(sp)[:,0]
            #fc = np.append(fc, np.int32(sp)[:,0], axis=0)
            #for i in sp:
            #    x, y = i[0]
            #    cv2.circle(fr, (int(x), int(y)), 2, (0, 255, 255), -1)
        fc = []
        #print(pts1)
        #tt = np.full(fr.shape, 255, dtype=np.uint8)
        tt = fr.copy()
        corner_msk = np.zeros(fr.shape[:-1], dtype=np.bool_)
        fmax = corners.max()
        BOX_SIZE=7
        for x, y in pts1:
            if x < -BOX_SIZE or x > fr.shape[1] + BOX_SIZE or y < -BOX_SIZE or y > fr.shape[0] + BOX_SIZE:
                #fc.append((x, y))
                continue
            #print(x, y)
            ix, iy = int(x), int(y)
            mx, my = max(ix - BOX_SIZE, 0), max(iy - BOX_SIZE, 0)
            region = corners[my:iy+BOX_SIZE,mx:ix+BOX_SIZE]
            #print(region)
            masked = (region > .0085 * fmax)#.002 * fmax)
            #corner_msk[my:iy+10,mx:ix+10] = masked#True
            tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,0] = masked * 255
            tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,1] = masked * 255
            tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,2] = masked * 255
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(masked.astype('uint8'))
            if len(centroids) <= 1:
                #fc.append((x, y))
                cv2.rectangle(cm, (ix - BOX_SIZE, iy - BOX_SIZE), (ix + BOX_SIZE, iy + BOX_SIZE), (0.0, 0.0, 1.0))
                continue
            cv2.rectangle(cm, (ix - BOX_SIZE, iy - BOX_SIZE), (ix + BOX_SIZE, iy + BOX_SIZE), (0.0, 1.0, 0.0))
            rc1 = np.log(region - (region.min() - 1))
            def key(x):
                i, (stat, centroid) = x
                i += 1
                rc = rc1.copy()
                rc[~(labels == i)] = 0
                return rc.sum()
            i, (st, (cx, cy)) = max(enumerate(zip(stats[1:], centroids[1:])), key=key)
            #print(cx, cy)
            cv2.circle(tt, (int(cx + mx), int(cy + my)), 1, (0, 0, 255), -1)
            #tt[int(cy+my),int(cx+mx)] = (0, 0, 255)
            fc.append((cx + mx, cy + my))
        #cv2.imshow('tt', tt)

        if False:
            eroded_hsv = cv2.dilate(eroded_hsv_1, kernel, iterations=2)
            contours, hier = cv2.findContours(eroded_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            biggest_c = itertools.islice(sorted(contours, key=cv2.contourArea, reverse=True), 2)
            zer = np.zeros(eroded_hsv.shape, np.uint8)
            for i in biggest_c:
                cv2.fillPoly(zer, pts=[i], color=255)
                cv2.drawContours(fr, i, -1, (255, 255, 255), 1)
            #corn2 = corners.copy()
            #corn2[~corner_msk] = 0
            #cactual = (corn2 > 0.1 * corn2.max()) & zer.astype(np.bool_) #eroded_hsv
            cactual = corner_msk & zer.astype(np.bool_)
            cv2.imshow('corner2', cactual.astype('uint8') * 255)

            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(cactual.astype('uint8'))
            #fr[eroded_hsv_1.astype(np.bool_),2] = 255
            #fr[cactual] = (0, 0, 255)
            fc = []
            for x, y in centroids[1:]:
                    #fr[int(y), int(x)] = (0, 255, 0)
                    ix, iy = int(x), int(y)
                    cc = eroded_hsv_1[iy-10:iy+10,ix-10:ix+10].astype(np.bool_).sum()

                    if 15 < cc < 160:
                        #cv2.putText(fr,str(cc),(ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                        #cv2.circle(fr, (int(x), int(y)), 2, (0, 255, 0), -1)
                        fc.append((x, y))
                    else: pass
                        #cv2.circle(fr, (int(x), int(y)), 2, (0, 0, 255), -1)
        c_exact = []
        #print(fc)
        if len(fc) > 0:
            #print(fc)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            c_exact = cv2.cornerSubPix(greenscale, np.float32(fc), (5, 5), (-1, -1), criteria)
            for x, y in c_exact:
                cv2.circle(fr, (int(x), int(y)), 3, (255, 64, 64), 1)
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
                    l = list(sorted(rect, key=key))
                    #for i, l1 in enumerate(l):
                    #    l2 = l[(i+1)%4]
                    #    cv2.line(fr, tuple(l1), tuple(l2), (0, 0, 255), 1, cv2.LINE_AA)
                    pts += l
                for i, (x, y) in enumerate(pts):
                    cv2.putText(fr,str(i),(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
                pts = np.float32(pts)
                #print(pts)
                retval2, rvecs, tvecs, inliers = cv2.solvePnPRansac(world_pts, pts, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
                #retval2, rvecs, tvecs, = cv2.solvePnP(world_pts, pts, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
                cv2.putText(fr,'I: {}'.format(len(inliers) if inliers is not None else 'X'),(550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255) if inliers is not None and len(inliers) == 8 else (0,0,255),1,cv2.LINE_AA)
                #retval2, rvecs, tvecs = cv2.solvePnP(world_pts, pts, cam_mtrx, distorts)
                #print(inliers)
                #print(rvecs, tvecs)
                #inliers = list(range(8))
                if inliers is not None:
                    o = set(range(8)) - set(inliers[:,0].tolist())
                    for idx in o:
                        x, y = pts[idx]
                        ix, iy = int(x), int(y)
                        #cv2.line(fr, (ix - 5, iy - 5), (ix + 5, iy + 5), (0, 0, 255), 1, cv2.LINE_AA)
                        #cv2.line(fr, (ix - 5, iy + 5), (ix + 5, iy - 5), (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.circle(fr, (ix, iy), 3, (0, 0, 255), 1)

                    centerp = np.float32([14.627/2, -5.325/2, 0])
                    global g_rot, g_pos
                    g_rot = rvecs
                    g_pos = tvecs
                    #ax2 = axis + centerp
                    #print(ax2)
                    #ax2 = np.insert(ax2, 0, centerp, axis=0)#np.float32([centerp]) + ax2
                    #print(ax2)
                    #imgpts, jac = cv2.projectPoints(ax2, rvecs, tvecs, cam_mtrx, distorts)
                    #fr =draw(fr, imgpts[0], imgpts[1:])

                    qb = cube + centerp
                    imgpts, jac = cv2.projectPoints(qb, rvecs, tvecs, cam_mtrx, distorts)
                    fr = draw_cube(fr, None, imgpts)

                    #pts2, jac = cv2.projectPoints(world_pts, rvecs, tvecs, cam_mtrx, distorts)
                    #for x, y in np.int32(pts2).reshape(-1, 2):
                    #    cv2.circle(fr, (x, y), 2, (255, 255, 255), -1)
            else:
                cv2.putText(fr,'C: {}'.format(len(c_exact)),(550, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)



        #pipeline.hsv_threshold_output
        #print(corners)
        #cv2.imshow('corners', corners)
        #cv2.imshow('corners', cm)
        cv2.imshow('dsst', op)
        cv2.imshow('f1', fr)
        #cv2.imshow('bc', zer)
        if cv2.waitKey(1) & 0xff == 27: break
    cv2.destroyAllWindows()
import threading
t = threading.Thread(target=main_loop)

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
        #self.scene = self.loader.loadModel("models/environment")
        self.scene = self.loader.loadModel("Target.egg")
        self.scene.setHpr(0, 90, 0)
        self.scene.reparentTo(self.render)
        #self.sph = self.loader.loadModel("smiley.egg")
        self.cam_ind = self.loader.loadModel("camera.egg")
        self.cam_ind.reparentTo(self.render)
        self.cam_ind.setScale(2)
        #self.disableMouse()
        self.useTrackball()
        print(self.trackball.node().getPos())
        #print(self.trackball.getNode(0))
        self.scene.setPos(0, 0, -.254)
        self.taskMgr.add(self.updateTask, "update")
        self.camLens.setFov(60)
    def updateTask(self, t):
        rod, jac = cv2.Rodrigues(g_rot)
        mat = np.append(np.append(rod, g_pos, axis=1), np.float32([[0, 0, 0, 1]]), axis=0)
        m2 = np.linalg.inv(mat)[:-1]
        tr2 = m2[:,3]
        rot2_rod = m2[:,:3]
        rot2, jac = cv2.Rodrigues(rot2_rod)
        x, y, z = tr2 * 2.54 #g_pos[:,0] * 2.54
        #rn, tn, *_ = cv2.composeRT(g_rot, g_pos, np.float32([[0, -math.pi, 0]]).T, np.float32([[0, 0, 0]]).T)
        rx, ry, rz = rot2[:,0] * (180 / math.pi)
        

        #print(rx, ry, rz)
        #print(x, y, z)
        #self.camera.setPos(-x, y, z)#*tvecs)
        self.cam_ind.setPos(x, z, -y)
        #self.camera.setPos(0, -50, 0)
        #ry, -rx, -rz
        #self.camera.setHpr(0, 0, 0)
        self.cam_ind.setHpr(-ry, rx, rz)#-rz, -rz)#rx, -rz)
        return Task.cont


 

if __name__ == '__main__':
    app = MyApp()
    t.start()
    app.run()
