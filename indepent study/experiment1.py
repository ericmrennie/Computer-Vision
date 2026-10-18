# canny edge detector inverted. White pixels with edges in black. Gives more of a drawing feel. 

#!/usr/bin/env python3
# reference - OpenCV API docs python 

import cv2 as cv
import numpy as np
vid = cv.VideoCapture(0) # default capture device

# ---------------------------------------------------------------- helpers
def to_gray(f):
    return cv.cvtColor(f, cv.COLOR_BGR2GRAY)

def to_bgr(g):
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)

def show_abs(x, alpha=1.0):
    """Derivative filters output signed values (negative = edge going the
    other way). Use a signed depth (CV_16S / CV_32F) to compute, then
    convertScaleAbs -> |x| * alpha, clipped to uint8 for display."""
    return to_bgr(cv.convertScaleAbs(x, alpha=alpha))

# ---------------------------------------------------------------- custom kernels via filter2D
# cv.filter2D(src, ddepth, kernel): ddepth=-1 means "same depth as src".
# NOTE: OpenCV computes correlation, not flipped convolution (matters only for asymmetric kernels).
# K_SHARPEN = np.array([[ 0, -1,  0],
#                       [-1,  5, -1],
#                       [ 0, -1,  0]], np.float32)

SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))  # also MORPH_RECT, MORPH_CROSS

while True:
  _, frame = vid.read()

  frame = frame[::3, ::3, :] # downsample (nearest) // takes every third pixel. image is 1/3x1/3 as big

  # frame = cv.filter2D(frame, -1, K_SHARPEN)

  frame = cv.Canny(to_gray(frame), 100, 200)
  frame = cv.bitwise_not(frame)


  cv.imshow('frame', frame) # posts frame on the screen

  if cv.waitKey(1) & 0xFF == ord('q'): # quits
    break

  if cv.waitKey(1) & 0xFF == ord(' '):
      cv.imwrite('/tmp/foo.jpg', frame) # saves a copy of frame

vid.release()
cv.destroyAllWindows()
