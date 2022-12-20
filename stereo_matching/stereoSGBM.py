# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

import cv2 as cv
import re
import numpy as np
from matplotlib import pyplot as plt
# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 9
min_disp = 0
max_disp = 320
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 0
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 0
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 0
disp12MaxDiff = 0

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
imgL = cv.imread('chess_l.png',0)
height = int(imgL.shape[0] * 0.5)
width = int(imgL.shape[1] * 0.5)
dim = (width, height)
imgL = cv.resize(imgL,dim) # scale the image down by a factor of 0.5
imgR = cv.imread('chess_r.png',0)
height = int(imgR.shape[0] * 0.5)
width = int(imgR.shape[1] * 0.5)
dim = (width, height)
imgR = cv.resize(imgR,dim)

disparity_SGBM = stereo.compute(imgL, imgR)
# Extract baseline, focal length from calibration file
with open('calib_chess2.txt', 'r') as file:
   for line in file:
      if re.search('cam0', line): focal_length = float(line[6:13])
      if re.search('baseline', line): baseline = float(line[9:])

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)

# Convert from disparity to depth
depth = baseline*focal_length/(disparity_SGBM+1e-9)
depth = depth.astype(np.uint8)

cv.imshow('Depth map using SGBM', disparity_SGBM)
cv.imwrite('chess_disparity_SGBM_9.png', disparity_SGBM)
