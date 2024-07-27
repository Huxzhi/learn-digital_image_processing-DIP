import cv2
import numpy as np


img_path = 'lena.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)


ret, thresh = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)

# & *
# *
#


def 腐蚀(img):
    h, w = img.shape
    dst = np.copy(img)
    for x in range(1, h-1):
        for y in range(1, w-1):
            dst[x, y] = 255 if img[x, y] + img[x, y+1] + img[x+1, y] else 0
    return dst


def 膨胀(img):
    h, w = img.shape
    dst = np.copy(img)
    for x in range(1, h-1):
        for y in range(1, w-1):
            dst[x, y] = 255 if img[x, y] * img[x, y+1] * img[x+1, y] else 0
    return dst


def 开运算(img):
    return 膨胀(腐蚀(img))


def 闭运算(img):
    return 腐蚀(膨胀(img))


cv2.imshow("imgGrey", thresh)
cv2.imshow("imgFushi", 腐蚀(thresh))
cv2.imshow("imgPengzhang", 膨胀(thresh))
cv2.imshow("imgKaiyunsuan", 开运算(thresh))
cv2.imshow("imgBiyunsuan", 闭运算(thresh))


cv2.waitKey()
