import cv2
import numpy as np
from progress.bar import Bar

def resize_img(source_img: np.array):
    shape_of_src = source_img.shape[:2]
    if max(shape_of_src) > 640:
        resize_coefficient = 640/max(shape_of_src)
        result_size = tuple([int(x*resize_coefficient) for x in reversed(shape_of_src)])
        result_img = cv2.resize(source_img, 
                                result_size,
                                interpolation=cv2.INTER_CUBIC)
        return result_img
    return source_img

def corr2(blockSize, shift, dev = True):
    img1 = resize_img(cv2.imread('left.png'))
    img2 = resize_img(cv2.imread('right.png'))
    blockimageL = np.copy(img1)
    blockimageR = np.copy(img2)

    width = img1.shape[1]
    height = img1.shape[0]
    baseline = 10
    focusline = 2507
    bf = baseline*focusline

    disparity = np.zeros((height, width), np.float32)
    disparity[::] = -1

    with Bar('Calculation', max=height-blockSize, suffix='%(percent)d%%') as bar:
        for i in range(height - blockSize):
            bar.next()
            for j in range(width - blockSize):
                x0 = j
                y0 = i

                img3 = img2[y0:y0 + blockSize + shift, x0:x0 + shift + blockSize]
                template = img1[y0:y0 + blockSize, x0:x0 + blockSize]

                h, w = blockSize, blockSize

                res = cv2.matchTemplate(img3, template, cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if (max_loc[0] == 0):
                    continue
                top_left = (max_loc[0] + j, max_loc[1] + i)
                bottom_right = (top_left[0] + w , top_left[1] + h)
                disparity[i, j] = bf / max_loc[0]
    return disparity

blocksize = 5
shift = int(blocksize*0.5)
print('blockSize = ' + str(blocksize))
print('shift = ' + str(shift))
print('Search zone: ' + str(blocksize+shift))
disp = corr2(blockSize = blocksize,
             shift = shift,
             dev = False)


disp_norm = (disp/disp.max())
#disp_norm = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR)
disp_norm = cv2.resize(disp_norm, (600, 600), interpolation=cv2.INTER_AREA)


cv2.imshow('res', disp_norm)
imgorig = cv2.resize(cv2.imread('left.png'), (600,600), interpolation=cv2.INTER_CUBIC)
cv2.imshow('orig', imgorig)


cv2.waitKey(0)
cv2.destroyAllWindows()
