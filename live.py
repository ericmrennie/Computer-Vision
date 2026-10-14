#!/usr/bin/env python3
# reference - OpenCV API docs python 

import cv2 as cv
vid = cv.VideoCapture(0) # default capture device

while True:
  _, frame = vid.read()

  frame = frame[::3, ::3, :] # downsample (nearest) // takes every third pixel. image is 1/3x1/3 as big
  #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # uncommented converts to greyscale
  #frame = cv.circle(frame, (447, 63), 23, (0, 0, 255), -1) # draws a circle on the frame

  frame = cv.medianBlur(frame, 9) # sorts pixels in 9x9 grid

  #frame = cv.putText(frame, \
  #        'Hello', \
  #        (50, 50), \
  #        cv.FONT_HERSHEY_SIMPLEX, \
  #        1, \
  #        (255, 0, 0), \
  #        2, \
  #        cv.LINE_AA)

  cv.imshow('frame', frame) # posts frame on the screen

  if cv.waitKey(1) & 0xFF == ord('q'): # quits
    break

  if cv.waitKey(1) & 0xFF == ord(' '):
      cv.imwrite('/tmp/foo.jpg', frame) # saves a copy of frame

vid.release()
cv.destroyAllWindows()

