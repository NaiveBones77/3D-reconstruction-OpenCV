import cv2
import numpy as np
from matplotlib import pyplot as plt

window_size = (600, 600)


def corr2(blockSize, shift, dev = True):
    img1 = cv2.imread('images/Bottle/im1.jpg')
    img2 = cv2.imread('images/Bottle/im0.jpg')
    width = img1.shape[1]
    height = img1.shape[0]
    scale = 0.3
    img1 = cv2.resize(img1, (int(scale*width), int(scale*height)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (int(scale*width), int(scale*height)), interpolation=cv2.INTER_AREA)
    width = img1.shape[1]
    height = img1.shape[0]
    doffs = 131.111
    blockimageL = np.copy(img1)
    blockimageR = np.copy(img2)

    baseline = 40
    focusline = 2507
    # baseline = 176.252e-3
    # focusline = 4152.073
    bf = baseline*focusline


    disparity = np.empty((height, width), np.float32)


    # for i in range(int(height/blockSize)):
    #     for j in range(int(width/blockSize)):
    for i in range(height - blockSize):
        for j in range(width - blockSize):
            x0 = j
            y0 = i
            img3 = img2[y0:y0 + blockSize + 20, x0:x0 + shift + blockSize]
            template = img1[y0:y0 + blockSize, x0:x0 + blockSize]

            h = template.shape[1]
            w = template.shape[0]

            res = cv2.matchTemplate(img3, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if (max_loc[0] == 0):
                continue

            top_left = (max_loc[0] + j, max_loc[1] + i)
            bottom_right = (top_left[0] + w , top_left[1] + h)
            disparity[i, j] = bf / (max_loc[0]  )

            if (dev):
                color = np.random.randint(0, 255, 3).tolist()
                cv2.imshow('img3', cv2.resize(img3, (300 + 2*shift, 320), interpolation=cv2.INTER_AREA))
                cv2.imshow('template', cv2.resize(template, (600, 600), interpolation=cv2.INTER_AREA))
                cv2.rectangle(blockimageR, top_left, bottom_right, (255, 0, 0), 2)
                cv2.rectangle(blockimageL, (x0, y0), (x0 + blockSize, y0 + blockSize), (255, 0, 0), 2)
                cv2.imshow('img1', blockimageL)
                cv2.imshow('finded', blockimageR)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(max_loc[0])
    return disparity



disp = corr2(blockSize = 10,
                 shift = 30,
             dev = False)

# last well worked result: bSize = 20, shift = 50 for Aloe.png


disp_norm = np.empty(disp.shape, dtype=np.float32)

cv2.normalize(disp, disp_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


# disp_norm = (disp/disp.max())
disp_norm = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR).astype(np.uint8)
disp_norm = cv2.resize(disp_norm, window_size, interpolation=cv2.INTER_AREA)


cv2.imshow('res', disp_norm)
imgorig = cv2.resize(cv2.imread('images/Bottle/im1.jpg'), window_size, interpolation=cv2.INTER_CUBIC)
cv2.imshow('orig', imgorig)

plt.imsave('Images/Disparities/BottleDisp.png', disp_norm)



cv2.waitKey(0)
cv2.destroyAllWindows()


