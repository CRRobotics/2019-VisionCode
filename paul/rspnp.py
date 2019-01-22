import ctypes
import numpy as np
import enum
import cv2
_dll = ctypes.CDLL('./librspnp.so')

float_p = ctypes.POINTER(ctypes.c_float)
double_p = ctypes.POINTER(ctypes.c_double)
int_p = ctypes.POINTER(ctypes.c_int)
_dll.solveRsPnP.argtypes = [ctypes.c_int, ctypes.c_int, float_p, float_p, float_p, float_p, double_p, double_p, double_p, double_p, ctypes.c_int, int_p, ctypes.c_bool, ctypes.c_int]
class SHUTTER(enum.Enum):
    GLOBAL = 0
    HORIZONTAL = 1
    VERTICAL = 2

def forceNpCast(v, typ):
    if isinstance(v, np.ndarray) and v.dtype == typ: return v
    return typ(v)
def np_to_ptr(x):
    if x is None: return None
    if x.dtype == np.float32:
        return ctypes.cast(x.ctypes.data, float_p)
    if x.dtype == np.float64:
        return ctypes.cast(x.ctypes.data, double_p)
    if x.dtype == np.int32:
        return ctypes.cast(x.ctypes.data, int_p)
def pbuf(x):
    return np.frombuffer(x, dtype=np.float64)[:,np.newaxis]
def solve_rspnp(objPoints, imgPoints, cameraMatrix, distCoeffs, shutterType, scanlines, rvec1=np.zeros((3, 1), dtype=np.float64), tvec1=np.zeros((3, 1), dtype=np.float64), useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE):
    assert len(objPoints) == len(imgPoints)
    objPoints = forceNpCast(objPoints, np.float32)
    imgPoints = forceNpCast(imgPoints, np.float32)
    cameraMatrix = forceNpCast(cameraMatrix, np.float32)
    assert len(scanlines) == 2
    scanlines = forceNpCast(scanlines, np.int32)
    if distCoeffs is not None: distCoeffs = forceNpCast(distCoeffs, np.float32)
    #rvec1 = np.empty((3, 1), dtype=np.float64)
    #tvec1 = np.empty((3, 1), dtype=np.float64)
    #rvec2 = np.empty((3, 1), dtype=np.float64)
    #tvec2 = np.empty((3, 1), dtype=np.float64)
    _rvec1 = ctypes.create_string_buffer(24)
    _tvec1 = ctypes.create_string_buffer(24)
    _rvec2 = ctypes.create_string_buffer(24)
    _tvec2 = ctypes.create_string_buffer(24)
    _rvec1[:] = rvec1.tobytes()
    _tvec1[:] = tvec1.tobytes()
    rv = _dll.solveRsPnP(len(objPoints), len(distCoeffs), np_to_ptr(objPoints), np_to_ptr(imgPoints), np_to_ptr(cameraMatrix), np_to_ptr(distCoeffs), ctypes.cast(_rvec1, double_p), ctypes.cast(_tvec1, double_p), ctypes.cast(_rvec2, double_p), ctypes.cast(_tvec2, double_p), shutterType.value, np_to_ptr(scanlines), useExtrinsicGuess, flags)
    if not rv: return False, None, None, None, None
    return True, pbuf(_rvec1), pbuf(_tvec1), pbuf(_rvec2), pbuf(_tvec2)

