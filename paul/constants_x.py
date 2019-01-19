import cv2
import numpy as np

        #self.__hsv_threshold_hue = [36.6546762589928, 102.07130730050935]
        #self.__hsv_threshold_saturation = [36.6906474820144, 255.0]
        #self.__hsv_threshold_value = [82.55395683453237, 255.0]

        # c270 new
        #self.__hsv_threshold_hue = [49, 139] 
        #self.__hsv_threshold_saturation = [140, 255.0]
        #self.__hsv_threshold_value = [76, 255.0]

        # c920
        #self.__hsv_threshold_hue = [49, 159]
        #self.__hsv_threshold_saturation = [92, 255.0]
        #self.__hsv_threshold_value = [44, 212]

        self.__hsv_threshold_hue = [70, 149] 
        self.__hsv_threshold_saturation = [46, 132]
        self.__hsv_threshold_value = [116, 255]


cam_mtrx =  np.array(
[[859.68420929,   0.,         318.24497556],
 [  0.,         891.25688996, 225.79027452],
 [  0.,           0.,           1.        ]], dtype='float32')

#cam_mtrx =  np.array(
#[[715,   0.,         310],
# [  0.,         715, 240],
# [  0.,           0.,           1.        ]], dtype='float32')

cam_mtrx_c920 = np.float32(
[[596.26209856,   0.,         453.26862804],
 [  0.,         580.57194811, 270.86714865],
 [  0.,           0.,           1.        ]])

distorts_c920 = np.float32([ 0.09042259, -0.26670032,  0.0015737,  -0.00970892,  0.2473531 ])


#targetColor = cv2.cvtColor(np.array([[[128, 255, 128]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
#targetColor = cv2.cvtColor(np.array([[[10 * 255 // 100, 47 * 255 // 100, 25 * 255 // 100]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
#targetColor = cv2.cvtColor(np.array([[[0x28, 0x68, 0x4e]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]
targetColor = cv2.cvtColor(np.array([[[0x69, 0x7f, 0xb0]]], 'uint8'), cv2.COLOR_RGB2LAB)[0,0]

