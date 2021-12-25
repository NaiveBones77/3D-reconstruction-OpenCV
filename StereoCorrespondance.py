import cv2
import numpy as np

window_size = 3
min_disp = 16
num_disp = 112 - min_disp

stereoSG = cv2.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=11,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=1,
                              uniquenessRatio=10,
                              speckleWindowSize=100,
                              speckleRange=32,
                              mode = 2)

stereo = cv2.StereoBM_create(num_disp, 11)


