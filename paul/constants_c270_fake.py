import cv2
import numpy as np

hsv_threshold_hue = [36.6546762589928, 102.07130730050935]
hsv_threshold_saturation = [36.6906474820144, 255.0]
hsv_threshold_value = [82.55395683453237, 255.0]

cam_mtrx =  np.array(
[[715,   0.,         310],
 [  0.,         715, 240],
 [  0.,           0.,           1.        ]], dtype='float32')

targetColor = cv2.cvtColor(np.array([[[128, 255, 128]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]

