import cv2
import numpy as np

hsv_threshold_hue = [8, 133]
hsv_threshold_saturation = [0, 255]
hsv_threshold_value = [140, 255]


cam_mtrx = np.float32(
[[596.26209856,   0.,         453.26862804],
 [  0.,         580.57194811, 270.86714865],
 [  0.,           0.,           1.        ]])

distorts = np.float32([ 0.09042259, -0.26670032,  0.0015737,  -0.00970892,  0.2473531 ])

targetColor = cv2.cvtColor(np.array([[[0xb1, 0xbc, 0xc5]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]

