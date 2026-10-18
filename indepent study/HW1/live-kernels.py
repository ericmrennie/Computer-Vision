#Claude generated 

#!/usr/bin/env python3
"""
live_kernels.py - your live.py template, extended with a catalog of OpenCV
image-processing primitives you can flip through while watching the webcam.

Keys:  n = next filter   p = previous filter   space = save frame   q = quit

Every filter is a function: BGR uint8 image in -> BGR uint8 image out.
So you can also import this file and call any of them on a still image:
    img = cv.imread('photo.jpg'); out = FILTERS['sobel magnitude'](img)
"""
import cv2 as cv
import numpy as np


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


# ---------------------------------------------------------------- smoothing
def identity(f):       return f
def box(f):            return cv.blur(f, (9, 9))                    # mean filter
def gaussian(f):       return cv.GaussianBlur(f, (9, 9), 0)         # sigma=0 -> derived from ksize
def gaussian_s3(f):    return cv.GaussianBlur(f, (0, 0), 3)         # ksize=(0,0) -> derived from sigma
def median(f):         return cv.medianBlur(f, 9)                   # non-linear; kills salt & pepper
def bilateral(f):      return cv.bilateralFilter(f, 9, 75, 75)      # edge-preserving blur


# ---------------------------------------------------------------- derivatives
# Pattern: gray -> (light blur to reduce noise) -> derivative at a signed depth -> abs
def _prep(f):
    return cv.GaussianBlur(to_gray(f), (3, 3), 0)

def sobel_x(f):   return show_abs(cv.Sobel(_prep(f), cv.CV_16S, 1, 0, ksize=3))  # vertical edges
def sobel_y(f):   return show_abs(cv.Sobel(_prep(f), cv.CV_16S, 0, 1, ksize=3))  # horizontal edges

def sobel_mag(f):
    g = _prep(f)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    return show_abs(cv.magnitude(gx, gy), alpha=0.5)   # sqrt(gx^2 + gy^2)

def scharr_x(f):  return show_abs(cv.Scharr(_prep(f), cv.CV_16S, 1, 0))          # better 3x3 rotational accuracy
def laplacian(f): return show_abs(cv.Laplacian(_prep(f), cv.CV_16S, ksize=3))    # 2nd derivative, direction-free
def canny(f):     return to_bgr(cv.Canny(to_gray(f), 100, 200))                  # multi-stage edge detector

def dog(f):
    """Difference of Gaussians ~ band-pass filter, approximates Laplacian of Gaussian."""
    g = to_gray(f)
    a = cv.GaussianBlur(g, (0, 0), 1).astype(np.int16)
    b = cv.GaussianBlur(g, (0, 0), 3).astype(np.int16)
    return show_abs(a - b, alpha=6)


# ---------------------------------------------------------------- custom kernels via filter2D
# cv.filter2D(src, ddepth, kernel): ddepth=-1 means "same depth as src".
# NOTE: OpenCV computes correlation, not flipped convolution (matters only for asymmetric kernels).
K_SHARPEN = np.array([[ 0, -1,  0],
                      [-1,  5, -1],
                      [ 0, -1,  0]], np.float32)
K_EMBOSS  = np.array([[-2, -1, 0],
                      [-1,  1, 1],
                      [ 0,  1, 2]], np.float32)
K_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], np.float32)
K_LAPLACE = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], np.float32)
K_BOX9    = np.ones((9, 9), np.float32) / 81.0

def sharpen(f):         return cv.filter2D(f, -1, K_SHARPEN)
def emboss(f):          return cv.filter2D(f, -1, K_EMBOSS)
def my_sobel_x(f):      return show_abs(cv.filter2D(to_gray(f), cv.CV_16S, K_SOBEL_X))  # same result as cv.Sobel
def unsharp(f):
    blur = cv.GaussianBlur(f, (0, 0), 3)
    return cv.addWeighted(f, 1.5, blur, -0.5, 0)       # f + 0.5*(f - blur)


# ---------------------------------------------------------------- morphology (structuring-element ops)
SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))  # also MORPH_RECT, MORPH_CROSS

def erode(f):           return cv.erode(f, SE)
def dilate(f):          return cv.dilate(f, SE)
def morph_open(f):      return cv.morphologyEx(f, cv.MORPH_OPEN, SE)       # erode then dilate (removes specks)
def morph_close(f):     return cv.morphologyEx(f, cv.MORPH_CLOSE, SE)      # dilate then erode (fills holes)
def morph_gradient(f):  return cv.morphologyEx(f, cv.MORPH_GRADIENT, SE)   # dilate - erode = outline
def tophat(f):          return cv.morphologyEx(f, cv.MORPH_TOPHAT, SE)     # src - open: small bright things
def blackhat(f):        return cv.morphologyEx(f, cv.MORPH_BLACKHAT, SE)   # close - src: small dark things


# ---------------------------------------------------------------- thresholding & intensity
def thresh_fixed(f):    return to_bgr(cv.threshold(to_gray(f), 127, 255, cv.THRESH_BINARY)[1])
def thresh_otsu(f):     return to_bgr(cv.threshold(to_gray(f), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1])
def thresh_adaptive(f): return to_bgr(cv.adaptiveThreshold(to_gray(f), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv.THRESH_BINARY, 11, 2))
def hist_eq(f):         return to_bgr(cv.equalizeHist(to_gray(f)))

_clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def clahe(f):           # local contrast, applied to lightness only so colors don't shift
    lab = cv.cvtColor(f, cv.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

def hsv_hue(f):         return cv.cvtColor(f, cv.COLOR_BGR2HSV)   # shows H,S,V channels as if they were B,G,R
def invert(f):          return cv.bitwise_not(f)


# ---------------------------------------------------------------- the menu
FILTERS = {
    'identity': identity,
    # smoothing
    'box blur': box, 'gaussian': gaussian, 'gaussian sigma=3': gaussian_s3,
    'median': median, 'bilateral': bilateral,
    # derivatives / edges
    'sobel x': sobel_x, 'sobel y': sobel_y, 'sobel magnitude': sobel_mag,
    'scharr x': scharr_x, 'laplacian': laplacian, 'canny': canny, 'difference of gaussians': dog,
    # custom kernels
    'sharpen (filter2D)': sharpen, 'emboss (filter2D)': emboss,
    'sobel x (hand-made kernel)': my_sobel_x, 'unsharp mask': unsharp,
    # morphology
    'erode': erode, 'dilate': dilate, 'open': morph_open, 'close': morph_close,
    'morph gradient': morph_gradient, 'top hat': tophat, 'black hat': blackhat,
    # thresholding / intensity
    'threshold fixed': thresh_fixed, 'threshold otsu': thresh_otsu,
    'threshold adaptive': thresh_adaptive, 'hist equalize': hist_eq, 'CLAHE': clahe,
    'HSV as BGR': hsv_hue, 'invert': invert,
}


def main():
    vid = cv.VideoCapture(0)                 # default capture device
    names = list(FILTERS)
    i = 0

    # peek at the actual numbers behind a Gaussian kernel (1-D; 2-D is its outer product)
    print('Gaussian 1-D kernel, ksize=9, sigma=2:\n', cv.getGaussianKernel(9, 2).ravel().round(3))
    print('Sobel 1-D pair (dx=1):', cv.getDerivKernels(1, 0, 3))

    while True:
        ok, frame = vid.read()
        if not ok:
            break
        frame = frame[::3, ::3, :]           # downsample: every third pixel

        name = names[i]
        out = FILTERS[name](frame)
        out = cv.putText(out.copy(), f'{i + 1}/{len(names)}  {name}', (8, 22),
                         cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('frame', out)

        key = cv.waitKey(1) & 0xFF           # call waitKey ONCE per loop; a 2nd call eats keypresses
        if key == ord('q'):
            break
        elif key == ord('n'):
            i = (i + 1) % len(names)
        elif key == ord('p'):
            i = (i - 1) % len(names)
        elif key == ord(' '):
            cv.imwrite('/tmp/foo.jpg', out)

    vid.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()