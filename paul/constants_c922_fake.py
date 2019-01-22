import cv2
import numpy as np
import math

hsv_threshold_hue = [39, 84]
hsv_threshold_saturation = [32, 255.0]
hsv_threshold_value = [53, 255]


cam_mtrx = np.float32(
[[596.26209856,   0.,         453.26862804 * 640 / 864],
 [  0.,         580.57194811, 270.86714865],
 [  0.,           0.,           1.        ]])
#foc_v = 2304 / 2 / math.tan(70.42 * math.pi / 180 / 2)
#foc_h = 1536 / 2 / math.tan(43.3 * math.pi / 180 / 2)
#cam_mtrx = np.float32(
#[[  foc_h * (480 / 1294),          0, 320.],
# [  0.,         foc_v * (640 / 1724), 240.],
# [  0.,                           0., 1.]])

distorts = np.float32([ 0.09042259, -0.26670032,  0.0015737,  -0.00970892,  0.2473531 ])

# 70.42° x 43.3°
#targetColor = cv2.cvtColor(np.array([[[10 * 255 // 100, 47 * 255 // 100, 25 * 255 // 100]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
targetColor = cv2.cvtColor(np.array([[[128, 255, 128]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
#resolution = (864, 480)

exposure = 130

# 1724 x 1294
