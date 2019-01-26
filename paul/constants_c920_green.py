import cv2
import numpy as np

hsv_threshold_hue = [49, 159]
hsv_threshold_saturation = [92, 255.0]
hsv_threshold_value = [44, 212]


#cam_mtrx = np.float32(
#[[596.26209856,   0.,         453.26862804],
# [  0.,         580.57194811, 270.86714865],
# [  0.,           0.,           1.        ]])
cam_mtrx = np.float32(
[[596.26209856,   0.,         453.26862804 * 640 / 864],
 [  0.,         580.57194811, 270.86714865],
 [  0.,           0.,           1.        ]])

distorts = np.float32([ 0.09042259, -0.26670032,  0.0015737,  -0.00970892,  0.2473531 ])

#targetColor = cv2.cvtColor(np.array([[[10 * 255 // 100, 47 * 255 // 100, 25 * 255 // 100]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
targetColor = cv2.cvtColor(np.array([[[130, 180, 165]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
#targetColor = cv2.cvtColor(np.array([[[0x28, 0x68, 0x4e]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]

