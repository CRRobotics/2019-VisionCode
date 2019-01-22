import cv2
import numpy as np

hsv_threshold_hue = [49, 139] 
hsv_threshold_saturation = [140, 255.0]
hsv_threshold_value = [76, 255.0]

cam_mtrx =  np.array(
[[715,   0.,         310],
 [  0.,         715, 240],
 [  0.,           0.,           1.        ]], dtype='float32')


distorts = None
targetColor = cv2.cvtColor(np.array([[[0x28, 0x68, 0x4e]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
exposure = 30
