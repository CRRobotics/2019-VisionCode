import cv2
import numpy
from scipy import ndimage
import math
from enum import Enum
import sys

# load constants from external file
constants = {}
s = exec(open(sys.argv[1], 'r').read(), constants)

_hsv_threshold_hue = constants['hsv_threshold_hue']
_hsv_threshold_saturation = constants['hsv_threshold_saturation']
_hsv_threshold_value = constants['hsv_threshold_value']

import os
import numpy as np
import sys

# parameters of the camera which can be determined using calibrate.py from OpenCV
cam_mtrx = constants['cam_mtrx']
distorts = constants['distorts']

# location of the target points, in world space, in order
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

# stolen from somewhere, draws a cube given perspective-transformed points
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
USE_CAP = True
world_pts = np.float32([(x, -y, 0) for x, y in world_pts])
print(world_pts)
print(cam_mtrx)
if USE_CAP:
    cap = cv2.VideoCapture(DEV)
    print('Got cap')
    exposure = constants['exposure'] if len(sys.argv) < 4 else int(sys.argv[3])
    os.system('v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c white_balance_temperature_auto=0 -c exposure_absolute={}'.format(DEV, exposure))
    os.system('v4l2-ctl -d /dev/video{} -c focus_auto=0'.format(DEV))
    os.system('v4l2-ctl -d /dev/video{} -c focus_absolute=0'.format(DEV))
    res_x, res_y = constants.get('resolution', (640, 480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)

targetColor = constants['targetColor']
lab_factors = 2, 1, 1
g_rot = np.float32([[0, 0, 0]]).T
g_pos = np.float32([[0, 0, 0]]).T
g_rot2 = np.float32([[0, 0, 0]]).T
g_pos2 = np.float32([[0, 0, 0]]).T
kernel = np.ones((5, 5), np.uint8)
from dataclasses import dataclass
import typing
@dataclass
            #return Rect(area=cv2.contourArea(lf), points=lf, refined=np.bool_(pointsRefined), center=(centerx, centery), tilt=ang)
class Rect:
    area: float
    points: typing.Any
    refined: np.ndarray
    center: np.ndarray
    v_vert: tuple
    tilt: float
import itertools
#import gil_load
#import rspnp
def main_loop():
    #gil_load.init()
    #gil_load.start()
    saves_left = 0
    while True:
        rv, fr = cap.read()
        if not rv: break
        #fr = cv2.imread('Target2.png')#.svg.png')

        fr_lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
        # convert frame to LAB color space
        channels = cv2.split(fr_lab.astype('int16'))

        # create image of how close each pixel is to a certain color
        greenscale = np.zeros(fr.shape[:-1], 'int16')
        for tr, ch, fac in zip(targetColor, channels, lab_factors):
            greenscale += np.absolute(ch - tr) // fac
        greenscale = 255 - np.clip((greenscale), 0, 255).astype('uint8')
        #greenscale = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        cv2.imshow('greenscale', greenscale)

        
        hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        op = cv2.inRange(hsv_frame, (_hsv_threshold_hue[0], _hsv_threshold_saturation[0], _hsv_threshold_value[0]),  (_hsv_threshold_hue[1], _hsv_threshold_saturation[1], _hsv_threshold_value[1]))

        o8 = op.astype('uint8') * 255
        corners = cv2.cornerHarris(np.float32(greenscale), 2, 3, 0.03)

        # version of corners image suitable for being displayed
        cm = corners.copy()
        cm[cm < 0] = 0
        cm = cv2.cvtColor((cm / 65536).astype(np.float32),cv2.COLOR_GRAY2RGB)
        
        # erode and dilate to reduce noise in thresholded image
        eroded_hsv_1 = cv2.dilate(cv2.erode(op, kernel, iterations=1), kernel, iterations=1)#.astype(np.bool_)

        # find contours
        contours1, hier1 = cv2.findContours(eroded_hsv_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        approxes = [cv2.approxPolyDP(x, 0.02 * cv2.arcLength(x, True), True) for x in contours1]

        a = False
        pts1 = []
        prevArea = 0
        rectangles = []
        LOOKAHEAD = 5
        kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        for area, loop in sorted(((cv2.contourArea(x), x) for x in contours1), key=lambda x: x[0], reverse=True):
            if len(loop) < 20: continue
            if area < prevArea / 2: continue
            prevArea = area
            cpts = []
            for i, v1 in enumerate(loop):
                v2 = loop[(i+LOOKAHEAD) % len(loop)]
                vec = np.float32((v2 - v1)[0])
                s = math.sqrt(sum(vec ** 2))
                if s == 0.0: continue
                #print(vec, s)
                vec /= s
                cpts.append(vec)
            cpts = np.float32(cpts)
            ret,label,center=cv2.kmeans(cpts,4,None,kmeans_criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            for i, v1 in enumerate(loop):
                v2 = loop[(i+LOOKAHEAD) % len(loop)]
                #print(v1[0], v2[0])
                #cm = cv2.line(cm, tuple(v1[0]), tuple(v2[0]), colors[label[i,0]], 1, cv2.LINE_AA) 

            ll = len(label)
            l3 = np.concatenate((label, label, label))
            kernel_1d = np.ones(LOOKAHEAD // 2, dtype=np.uint8)

            sides = []
            for i in range(len(center)):
                f = cv2.erode((l3 == i).astype(np.uint8), kernel_1d)[ll:ll*2,0].astype(np.bool_)
                #print(f.T)
                #idxs = f | np.roll(f, LOOKAHEAD)#np.concatenate((f, (f + LOOKAHEAD) % len(loop)))
                idxs = np.roll(f, LOOKAHEAD // 2)
                s_pts = loop[idxs]
                #print(s_pts)
                if len(s_pts) == 0: break # continue
                vx, vy, xi, yi = cv2.fitLine(s_pts, cv2.DIST_L2, 0, 0.01, 0.01)[:,0]
                sz = sum(idxs)
                sides.append((sz, (vx, vy, xi, yi)))

                #print(vx, vy, xi, yi)
                #fr = cv2.line(fr, (int(xi + vx * -50), int(yi + vy * -50)), (int(xi + vx * 50), int(yi + vy * 50)), (255, 255, 255), 1, cv2.LINE_AA)
                #rv, labels, stats, centroids = cv.connectedComponentsWithStats(f)
            if len(sides) < 4: continue
            sd = list(sorted(sides, key=lambda x: x[0]))
            order = sd[0], sd[2], sd[1], sd[3]
            corners_ = []
            for i, s1 in enumerate(order):
                s2 = order[(i+1)%4]
                vx1, vy1, ix1, iy1 = s1[1]
                vx2, vy2, ix2, iy2 = s2[1]
                try:
                    res = np.linalg.solve(np.float32([
                        [vx1, 0, -1, 0],
                        [vy1, 0, 0, -1],
                        [0, vx2, -1, 0],
                        [0, vy2, 0, -1]]), np.float32([[-ix1, -iy1, -ix2, -iy2]]).T)
                except np.linalg.LinAlgError:
                    break
                #print(res[0,2:4,0])
                corners_.append(res[2:4,0])
            if len(corners_) < 4: continue
            cc = np.float32(corners_)
            #fr = cv2.drawContours(fr, [cc.astype('int32')], -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            rectangles.append(cc)
            #print(corners_)
            #for x, y in corners_:
            #    if 0 < x < fr.shape[1] and 0 < y < fr.shape[0]: cm[int(y), int(x)] = (0, 0, 240)

            #print(label, center)


        ## Find contour edges, then find their intersections to find approximate corner locations
        # iterate over contours, large to small
        #cm = cv2.drawContours(cm, approxes,-1,(0,0,255),1)

        # Look for Harris corners in the areas selected by edge intersections
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        fc = []
        onscreen_fc = []
        #tt = fr.copy()
        fmax = corners.max()
        BOX_SIZE=7
        def refine_point(x, y):
            ix, iy = int(x), int(y)
            if x < -BOX_SIZE /2  or x > fr.shape[1] + BOX_SIZE / 2 or y < -BOX_SIZE / 2 or y > fr.shape[0] + BOX_SIZE / 2:
                cv2.rectangle(cm, (ix - BOX_SIZE, iy - BOX_SIZE), (ix + BOX_SIZE, iy + BOX_SIZE), (1.0, 0.0, 0.0))
                return (x, y), False
            # find coords of top-left corner
            mx, my = max(ix - BOX_SIZE, 0), max(iy - BOX_SIZE, 0)
            region = corners[my:iy+BOX_SIZE,mx:ix+BOX_SIZE]
            masked = (region > .1 * region.max())#.002 * fmax)
            # update debug image
            #tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,0] = masked * 255
            #tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,1] = masked * 255
            #tt[my:iy+BOX_SIZE,mx:ix+BOX_SIZE,2] = masked * 255
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(masked.astype('uint8'))
            if len(centroids) <= 1:
                #if x < 0 or x > fr.shape[1] or y < 0 or y > fr.shape[0]:
                cv2.rectangle(cm, (ix - BOX_SIZE, iy - BOX_SIZE), (ix + BOX_SIZE, iy + BOX_SIZE), (0.0, 0.0, 1.0))
                return (x, y), False
                #fc.append((x, y))
            cv2.rectangle(cm, (ix - BOX_SIZE, iy - BOX_SIZE), (ix + BOX_SIZE, iy + BOX_SIZE), (0.0, 1.0, 0.0))
            rc1 = np.log(region - (region.min() - 1))
            def key(x):
                i, (stat, centroid) = x
                #rc = rc1.copy()
                #rc[labels != (i + 1)] = 0
                return rc1[labels == (i+1)].sum()
            # get connected component with the highest total Harris corner value
            i, (st, (cx, cy)) = max(enumerate(zip(stats[1:], centroids[1:])), key=key)
            #cv2.circle(tt, (int(cx + mx), int(cy + my)), 1, (0, 0, 255), -1)
            rcc = rc1.copy()
            rcc[labels != (i+1)] = 0
            cx2, cy2 = ndimage.measurements.center_of_mass(rcc)
            c_exact1 = cv2.cornerSubPix(greenscale, np.float32([[cx2 + mx, cy2 + my]]), (5, 5), (-1, -1), criteria)
            return c_exact1[0], True

        def create_Rect(l):
            points, pointsRefined = zip(*l)
            pairs = [(points[i], points[(i+1)%len(points)]) for i in range(len(points))]
            #for p1, p2 in pairs:
            #    cv2.line(fr, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 240), 1, cv2.LINE_AA) 
            #for p1 in points:
            #    if 0 < p1[0] < fr.shape[1] and 0 < p1[1] < fr.shape[0]: fr[int(p1[1]), int(p1[0])] = (0, 0, 240)
            relevantLines = list(sorted(pairs, key=lambda x: ((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2), reverse=True))
            r2 = list(sorted(relevantLines[2:4], key=lambda x: min(x[0][1], x[1][1]), reverse=True))
            #print(len(relevantLines))
            #def ap(a, b):
            #    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            pbot = np.sum(r2[0], axis=0) / 2
            ptop = np.sum(r2[1], axis=0) / 2
            ang = math.atan2(ptop[1] - pbot[1], ptop[0] - pbot[0]) % math.pi
            #cv2.putText(fr,'{:.2f}'.format(ang),(int(x1 + x2) // 2,int(y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
            #print(ang)
            centerx, centery = np.mean(np.float32(points), axis=0)
            def key(r):
                (x, y), legit = r
                dx, dy = x - centerx, y - centery
                return math.atan2(-dy, dx) % (2 * math.pi)
            points, pointsRefined = zip(*sorted(l, key=key))
            lf = np.float32(points)
            return Rect(area=cv2.contourArea(lf), points=lf, refined=np.bool_(pointsRefined), center=np.float32([centerx, centery]), tilt=ang, v_vert=(pbot, ptop))
        refined_rects = [create_Rect([refine_point(x, y) for x, y in rect]) for rect in rectangles]# if len(rect) == 4]
        refined_rects = [x for x in refined_rects if x.refined.any()]
        for r in refined_rects:
            fr = cv2.drawContours(fr, [r.points.astype('int32')], -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        pairings = []
        #for r1 in refined_rects:
        i = 0
        def coord_change(pt, v1, v2):
            return np.linalg.solve(np.float32(
                [[v1[0], v2[0]],
                 [v1[1], v2[1]]]), np.float32(pt).T).T
        def rot(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array(((c,-s), (s, c)))
        r_left = rot(np.radians(-14.5))
        r_right = rot(np.radians(14.5))
        while i < len(refined_rects):
            r1_ = refined_rects[i]
            def dst(r2):
                #r2 = r2[1]
                pbot, ptop = r2.v_vert
                vec = ptop - pbot
                nvec = vec / np.sqrt(sum(vec ** 2))
                pvec = nvec[[1,0]] * [1, -1]
                rvec = r2.center - r1_.center
                xf, yf = coord_change(rvec, pvec, nvec)
                x1, y1 = coord_change(np.dot(rvec, r_right if xf > 0 else r_left), np.dot(pvec, r_right if xf > 0 else r_left), nvec)
                return x1 ** 2 + y1 ** 2 * 10
            for j, (r2_, dst) in sorted(enumerate((x, dst(x)) for x in refined_rects[i+1:]), key=lambda x: x[1][1]):#lambda r: sum((r[1].center - r1_.center) ** 2)):
                j += i + 1
                #if r2_ is r1_: continue
                r1, r2 = (r1_, r2_) if r1_.center[0] < r2_.center[0] else (r2_, r1_)
                #if r2.center[0] < r1.center[0]: continue
                ltr_vec = r2.center - r1.center
                ltr = math.atan2(-ltr_vec[1], ltr_vec[0])
                def rect_sort(rect):
                    def key(i):
                        pt = rect.points[i]
                        v = pt - rect.center
                        return (math.atan2(-v[1], v[0]) - ltr) % (2 * math.pi)
                    l = list(sorted(list(range(len(rect.points))), key=key))
                    return rect.points[l], rect.refined[l]#, l
                ls, ls_r = rect_sort(r1)
                rs, rs_r = rect_sort(r2)
                p1 = np.append(ls, rs, axis=0)
                v1 = np.append(ls_r, rs_r, axis=0)
                if not v1[[0,1,4,5]].any() or not v1[[2,3,6,7]].any(): continue
                if sum(v1) < 4: continue
                #rps = np.append(li, ri + 4, axis=0)
                p2 = p1[v1]
                wps = world_pts[v1]
                #for q, (xx, yy) in enumerate(p2):
                #    cv2.putText(fr,str(q),(int(xx),int(yy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                #print(v1, wps, p2)
                try:
                    #retval2_, rvecs, tvecs, inliers_ = cv2.solvePnPRansac(wps, p2, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
                    #print(p2)
                    #print(p2)
                    p3 = p2.reshape(-1,1,2)#p2[:,np.newaxis,:]
                    #print(p3)
                    wp3 = wps.reshape(-1,1,3)
                    rvecs = np.empty((3, 1), dtype=np.float32)
                    tvecs = np.empty((3, 1), dtype=np.float32)
                    retval2_, rvecs, tvecs = cv2.solvePnP(wp3, p3, cam_mtrx, distorts, rvecs, tvecs, flags=cv2.SOLVEPNP_EPNP, useExtrinsicGuess=False)
                    #print('R')
                    #print(rvecs)
                    #print('T')
                    #print(tvecs)
                except cv2.error as e:
                    pass
                    print(e)
                else:
                    if retval2_:
                        imgpts, jac = cv2.projectPoints(wps, rvecs, tvecs, cam_mtrx, distorts)
                        #print(p2)
                        #print(imgpts[0,:])
                        imgpts = imgpts[:,0]
                        sp = np.stack((p2, imgpts), axis=1)
                        st = (p2 - imgpts) ** 2
                        #print(st)
                        err = np.sum(np.sqrt(np.sum(st, axis=1))) / len(p2) / math.sqrt(cv2.contourArea(r1.points) + cv2.contourArea(r2.points))
                        #print(err)
                        #err = sum(sum((a - b) ** 2) for a, b in zip(p2, imgpts)) / len(p2) / (cv2.contourArea(r1.points) + cv2.contourArea(r2.points))#sum(ltr_vec ** 2)
                        #for a, b in zip(p2, imgpts):
                        #    print(a, b, (a - b), (a - b) ** 2, sum((a - b) ** 2))
                        #print(err)
                        if err < .1:
                            if (r1.center > 0).all() and (r2.center > 0).all(): cv2.line(fr, tuple(map(int, r1.center)), tuple(map(int, r2.center)), (32, 255, 0), 1, cv2.LINE_AA)
                            #centerp = (r1.center + r2.center) / 2
                            pairings.append((err, (r1, r2), (rvecs, tvecs), (p2, wps)))
                            del refined_rects[j]
                            break

            i += 1
        for h in pairings:
        #if False:
            er, (r1, r2), (r, t), (ip, wp) = h
            if len(ip) < 6: continue
            retval2_, rvecs, tvecs, inliers = cv2.solvePnPRansac(wp, ip, cam_mtrx, distorts, iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
            centerp = np.float32([14.627/2, -5.325/2, 0])
            qb = cube + centerp
            #print(qb)
            imgpts, jac = cv2.projectPoints(qb, rvecs, tvecs, cam_mtrx, distorts)
            #print(cam_mtrx)
            #print(distorts)
            if not any(abs(x) > 1500 for x in imgpts.flatten()):
                fr = draw_cube(fr, None, imgpts)
                xx, yy = imgpts[2,0]
                cv2.putText(fr,'{:.1f} {:.1f} {:.1f}'.format(*tvecs[:,0]),(int(xx),int(yy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                #cv2.putText(fr,'{:.3e}'.format(er),(int(xx),int(yy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                #cv2.putText(fr,'{} {:.3e}'.format(len(inliers), err),(int(xx),int(yy + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)


        #valids = list(sorted(pairings, key=lambda x: x[0]))[:len(refined_rects)//2]
        #for i in valids:
        #    r1, r2 = i[1]
        #    cv2.line(fr, tuple(map(int, r1.center)), tuple(map(int, r2.center)), (0, 0, 255), 1, cv2.LINE_AA)
            

        
        OFFS = 14.5 * math.pi / 180
            
        c_exact = []
        if any(onscreen_fc):
            # refine corners to sub-pixel level
            c_exact1 = cv2.cornerSubPix(greenscale, np.float32([x for i, x in enumerate(fc) if onscreen_fc[i]]), (5, 5), (-1, -1), criteria)
            #if len(c_exact1) < len(onscreen_fc): continue
            c_exact = []
            j = 0
            for i, v in enumerate(onscreen_fc):
                if v:
                    c_exact.append((c_exact1[j], True))
                    j += 1
                else:
                    c_exact.append((fc[i], False))
            for (x, y), legit in c_exact:
                try:
                    cv2.circle(fr, (int(x), int(y)), 3, (255, 64, 64), 1)
                except OverflowError:
                    pass
            #for i, (x, y, z) in enumerate(world_pts):
            #    cv2.putText(fr,str(i),(int(x*5),int(y*5)+50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
            
            ## Sort points. First split left and right rectangle, then sort counterclockwise by angle to the center
            if sum(onscreen_fc) > 4 and len(onscreen_fc) == 8:
                h = list(sorted(c_exact, key=lambda x: x[0][0]))
                left = h[:4]
                right = h[4:]
                pts = []
                for rect in left, right:
                    centerx, centery = sum(x[0][0] for x in rect) / len(rect), sum(x[0][1] for x in rect) / len(rect)
                    def key(r):
                        (x, y), legit = r
                        dx, dy = x - centerx, y - centery
                        return math.atan2(-dy, dx) % (2 * math.pi)
                    l = list(sorted(rect, key=key))
                    #for i, l1 in enumerate(l):
                    #    l2 = l[(i+1)%4]
                    #    cv2.line(fr, tuple(l1), tuple(l2), (0, 0, 255), 1, cv2.LINE_AA)
                    pts += l
                for i, ((x, y), legit) in enumerate(pts):
                    cv2.putText(fr,str(i),(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
                pts_to_solve = np.bool_(onscreen_fc)
                fpts = np.float32([x[0] for x in pts])
                pts_f = fpts[pts_to_solve]
                world_pts_f = world_pts[pts_to_solve]
                global g_rot, g_pos, g_rot2, g_pos2
                if not ((g_rot == 0).all() and (g_pos == 0).all()):
                    retval2_, rvecs, tvecs, inliers_ = cv2.solvePnPRansac(world_pts_f, pts_f, cam_mtrx, distorts, g_rot, g_pos, useExtrinsicGuess=True)#, flags=cv2.SOLVEPNP_EPNP)
                else:
                    retval2_, rvecs, tvecs, inliers_ = cv2.solvePnPRansac(world_pts_f, pts_f, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
                #print(world_pts_f)
                #print(pts_f)
                #if not ((g_rot == 0).all() and (g_pos == 0).all()):
                #    retval2, rvecs1, tvecs1, rvecs2, tvecs2 = rspnp.solve_rspnp(world_pts_f, pts_f, cam_mtrx, distorts, rspnp.SHUTTER.VERTICAL, [0, 480], rvec1=g_rot2, tvec1=g_pos2, useExtrinsicGuess=True)#, flags=cv2.SOLVEPNP_EPNP)
                #else:
                #    retval2, rvecs1, tvecs1, rvecs2, tvecs2 = rspnp.solve_rspnp(world_pts_f, pts_f, cam_mtrx, distorts, rspnp.SHUTTER.VERTICAL, [0, 480])#, flags=cv2.SOLVEPNP_EPNP)
                #rvecs_ = (rvecs1 + rvecs2) / 2
                #tvecs_ = (tvecs1 + tvecs2) / 2
                print(np.stack((rvecs[:,0], tvecs[:,0]), axis=1))
                #print(np.stack((rvecs_[:,0], tvecs_[:,0]), axis=1))
                #print(rvecs, tvecs, rvecs2, tvecs2)
                inliers = np.expand_dims(np.arange(8), axis=1)
                #retval2, rvecs, tvecs, = cv2.solvePnP(world_pts, pts, cam_mtrx, distorts)#, flags=cv2.SOLVEPNP_EPNP)
                cv2.putText(fr,'I: {}/{}'.format(len(inliers) if inliers is not None else 'X', len(pts_f)),(530, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255) if inliers is not None and len(inliers) == len(pts_f) else (0,0,255),1,cv2.LINE_AA)
                if inliers is not None:
                    o = set(range(len(pts_f))) - set(inliers[:,0])
                    for idx in o:
                        x, y = pts_f[idx]
                        ix, iy = int(x), int(y)
                        #cv2.line(fr, (ix - 5, iy - 5), (ix + 5, iy + 5), (0, 0, 255), 1, cv2.LINE_AA)
                        #cv2.line(fr, (ix - 5, iy + 5), (ix + 5, iy - 5), (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.circle(fr, (ix, iy), 3, (0, 0, 255), 1)

                    centerp = np.float32([14.627/2, -5.325/2, 0])
                    g_rot = rvecs
                    g_pos = tvecs
                    #g_rot2 = rvecs_
                    #g_pos2 = tvecs_

                    # Project cube according to perspective and display
                    qb = cube + centerp
                    #print(qb)
                    imgpts, jac = cv2.projectPoints(qb, rvecs, tvecs, cam_mtrx, distorts)
                    #print(cam_mtrx)
                    #print(distorts)
                    #print(imgpts)
                    if not any(abs(x) > 1500 for x in imgpts.flatten()):
                        fr = draw_cube(fr, None, imgpts)

                    #pts2, jac = cv2.projectPoints(world_pts, rvecs, tvecs, cam_mtrx, distorts)
                    #for x, y in np.int32(pts2).reshape(-1, 2):
                    #    cv2.circle(fr, (x, y), 2, (255, 255, 255), -1)
            else:
                cv2.putText(fr,'C: {}'.format(len(c_exact)),(550, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)



        #cv2.imshow('corners', corners)
        cv2.imshow('corners', cm)
        cv2.imshow('dsst', op)
        #cv2.imshow('tt', tt)
        cv2.imshow('f1', fr)
        if saves_left > 0:
            cv2.imwrite('frame_{}.png'.format(10 - saves_left), fr)
            saves_left -= 1
        #cv2.imshow('bc', zer)
        key = cv2.waitKey(1) & 0xff
        if key == 27: break
        if key == ord('s') and saves_left == 0:
            saves_left = 10
    cv2.destroyAllWindows()
    #gil_load.stop()
    #print(gil_load.get(N=4))

#import threading
#t = threading.Thread(target=main_loop)

#from direct.showbase.ShowBase import ShowBase
#from direct.task import Task
def invertPerspective(rvecs, tvecs):
        rod, jac = cv2.Rodrigues(rvecs)
        mat = np.append(np.append(rod, tvecs, axis=1), np.float32([[0, 0, 0, 1]]), axis=0)
        m2 = np.linalg.inv(mat)[:-1]
        tr2 = m2[:,3]
        rot2_rod = m2[:,:3]
        rot2, jac = cv2.Rodrigues(rot2_rod)
        return rot2, tr2

#class MyApp(ShowBase):
class MyApp:
    def __init__(self):
        ShowBase.__init__(self)
        #self.scene = self.loader.loadModel("models/environment")
        self.scene = self.loader.loadModel("Target.egg")
        self.scene.setHpr(0, 90, 0)
        self.scene.reparentTo(self.render)
        #self.sph = self.loader.loadModel("smiley.egg")
        self.cam_ind = self.loader.loadModel("camera.egg")
        self.cam_ind.reparentTo(self.render)
        self.cam_ind.setScale(2.5)
        self.cam_ind.setColorScale(1.0, 0.6, 0.6, 1.0)
        self.cam_ind2 = self.loader.loadModel("camera.egg")
        self.cam_ind2.reparentTo(self.render)
        self.cam_ind2.setScale(2.5)
        self.cam_ind2.setColorScale(0.6, 0.6, 1.0, 1.0)
        self.useTrackball()
        print(self.trackball.node().getPos())
        self.scene.setPos(0, 0, -.254)
        self.taskMgr.add(self.updateTask, "update")
        self.camLens.setFov(60)
    def updateTask(self, t):
        # Find inverse of perspective transform (get "camera moves" perspective)
        persp1 = invertPerspective(g_rot, g_pos)
        #persp2 =invertPerspective(g_rot2, g_pos2)
        #rod, jac = cv2.Rodrigues(g_rot)
        #mat = np.append(np.append(rod, g_pos, axis=1), np.float32([[0, 0, 0, 1]]), axis=0)
        #m2 = np.linalg.inv(mat)[:-1]
        #tr2 = m2[:,3]
        #rot2_rod = m2[:,:3]
        #rot2, jac = cv2.Rodrigues(rot2_rod)
        # 2.54 cm/inch
        for cm, (rot2, tr2) in ((self.cam_ind, persp1),):# (self.cam_ind2, persp2)):
            if np.isnan(rot2).any() or np.isnan(tr2).any(): continue
            x, y, z = tr2 * 2.54 #g_pos[:,0] * 2.54
            rx, ry, rz = rot2[:,0] * (180 / math.pi)
            #print(id(cm), x, y, z)
            cm.setPos(x, z, -y)
            cm.setHpr(-ry, rx, rz)#-rz, -rz)#rx, -rz)
        return Task.cont

if __name__ == '__main__':
    main_loop()
    #app = MyApp()
    #t.start()
    #app.run()
