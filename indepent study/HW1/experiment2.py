# Laplacian of Gaussian with a 9x9 Gaussian kernel and a Laplacian scale of 7. 
# Gives topographic visuals

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

while True:
  _, frame = vid.read()

  frame = frame[::3, ::3, :] # downsample (nearest) // takes every third pixel. image is 1/3x1/3 as big
  
  frame = cv.GaussianBlur(to_gray(frame), (9, 9), 0)
  frame = show_abs(cv.Laplacian(frame, cv.CV_16S, ksize=7))
  cv.imshow('frame', frame) # posts frame on the screen

  if cv.waitKey(1) & 0xFF == ord('q'): # quits
    break

  if cv.waitKey(1) & 0xFF == ord(' '):
      cv.imwrite('/tmp/foo.jpg', frame) # saves a copy of frame

vid.release()
cv.destroyAllWindows()
