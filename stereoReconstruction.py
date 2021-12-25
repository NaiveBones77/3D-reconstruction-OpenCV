import cv2
import configparser
import glob
import numpy as np
from matplotlib import pyplot as plt


def getDisparity(save = False):
    config = configparser.ConfigParser()
    config.read(glob.glob('config.conf')[0])

    imgL = cv2.imread(config.get('disparity_config', 'pathL'), cv2.CV_8UC1)
    imgR = cv2.imread(config.get('disparity_config', 'pathR'), cv2.CV_8UC1)

    stereo = cv2.StereoBM_create(16*12, 21)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0


    disp_norm = np.empty(disparity.shape, np.float32)
    disp_norm = cv2.normalize(disparity, disp_norm, 1, 255, cv2.NORM_MINMAX)
    max = disp_norm.max()
    disp_norm = 1 / disp_norm

    (h, w) = disparity.shape
    # center = (h/2, w/2)
    # M = cv2.getRotationMatrix2D(center, 90, 1.0)
    # disparity = cv2.warpAffine(disparity, M, (h, w))

    if (save):
        imgsave = cv2.resize(disparity, (600,600), interpolation=cv2.INTER_AREA)
        cv2.imwrite('images/Disparities/ChairStereoDisp.png', imgsave)


    plt.imshow(disp_norm)
    plt.show()
    cv2.imshow('disp', disp_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

getDisparity(True)

