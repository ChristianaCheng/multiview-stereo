import cv2 as cv
import re
import numpy as np
from matplotlib import pyplot as plt
# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = 0
max_disp = 320
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 15
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

stereo = cv.StereoBM_create(numDisparities=num_disp,blockSize=block_size)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)
stereo.setMinDisparity(min_disp)

imgL = cv.imread('chess_l.png',0)
imgR = cv.imread('chess_r.png',0)

disparity_BM = stereo.compute(imgL, imgR)
# Extract baseline, focal length from calibration file
with open('calib_chess2.txt', 'r') as file:
   for line in file:
      if re.search('cam0', line): focal_length = float(line[6:13])
      if re.search('baseline', line): baseline = float(line[9:])

# Normalize the values to a range from 0..255 for a grayscale image
disparity_BM = cv.normalize(disparity_BM, disparity_BM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_BM = np.uint8(disparity_BM)

# Convert from disparity to depth
depth = baseline*focal_length/(disparity_BM+1e-9)
depth = depth.astype(np.uint8)

plt.imshow(depth, cmap='viridis')
plt.colorbar()
plt.savefig('depth_BM_norm.png')
plt.show()
