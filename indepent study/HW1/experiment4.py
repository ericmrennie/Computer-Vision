# Erode 75x75 cross with HSV. Creates large colorful crosses. 
# This is another iteration of experiment3. Returns warmer colors.

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

SE = cv.getStructuringElement(cv.MORPH_CROSS, (75, 75))

while True:
  _, frame = vid.read()

  frame = frame[::3, ::3, :] # downsample (nearest) // takes every third pixel. image is 1/3x1/3 as big
  
  frame = cv.dilate(frame, SE)
  frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
  cv.imshow('frame', frame) # posts frame on the screen

  if cv.waitKey(1) & 0xFF == ord('q'): # quits
    break

  if cv.waitKey(1) & 0xFF == ord(' '):
      cv.imwrite('/tmp/foo.jpg', frame) # saves a copy of frame

vid.release()
cv.destroyAllWindows()
