import numpy as np
import cv2

hsv_threshold_hue = [70, 149] 
hsv_threshold_saturation = [46, 132]
hsv_threshold_value = [116, 255]

cam_mtrx =  np.array(
[[715,   0.,         310],
 [  0.,         715, 240],
 [  0.,           0.,           1.        ]], dtype='float32')

distorts = None
#targetColor = cv2.cvtColor(np.array([[[0x69, 0x7f, 0xb0]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
targetColor = cv2.cvtColor(np.array([[[0x89, 0x9f, 0xd0]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
