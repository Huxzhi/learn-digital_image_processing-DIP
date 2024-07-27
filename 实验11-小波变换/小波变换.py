import cv2
import numpy as np


img_path = 'lena.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)


ret, thresh = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)


def 行小波变换(img):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(1, int(w/2)):
            dst[i, j] = img[i, 2*j]
            dst[i, int(j+w/2)] = img[i, 2*j] - img[i, 2*j - 1] + 128
    return np.uint8(dst)


def 列小波变换(img):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, int(h/2)):
        for j in range(w):
            dst[i, j] = img[2*i, j]
            dst[int(i+h/2), j] = img[2*i, j]-img[2*i-1,  j] + 128
    return np.uint8(dst)


def 低通滤波(img):
    h, w = img.shape
    dst = np.full((h, w), 128, dtype=np.uint8)
    for i in range(int(h/2)):
        for j in range(int(w/2)):
            dst[i, j] = img[i, j]

    return np.uint8(dst)


def 高通滤波(img):
    h, w = img.shape
    dst = np.copy(img)
    for i in range(int(h/2)):
        for j in range(int(w/2)):
            dst[i, j] = 128

    return np.uint8(dst)


def 逆行小波变换(img):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(1, int(w/2)):
            dst[i, j*2] = img[i, j]
            dst[i, 2*j - 1] = img[i, j] - img[i, j + int(w/2)] + 128
    return np.uint8(dst)


def 逆列小波变换(img):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, int(h/2)):
        for j in range(w):
            dst[i*2, j] = img[i, j]
            dst[i*2 - 1, j] = img[i, j] - img[i + int(h/2), j] + 128
    return np.uint8(dst)


cv2.imshow("imgGrey", imgGrey)

小波变换 = 列小波变换(行小波变换(imgGrey))
cv2.imshow("imgGrey1", 小波变换)

cv2.imshow("imgGrey2",  逆列小波变换(逆行小波变换(小波变换)))

小波低通 = 低通滤波(小波变换)
cv2.imshow("imgGrey3", 小波低通)
小波高通 = 高通滤波(小波变换)
cv2.imshow("imgGrey4", 小波高通)

cv2.imshow("imgGrey5",  逆列小波变换(逆行小波变换(小波高通)))
cv2.imshow("imgGrey6", 逆列小波变换(逆行小波变换(小波低通)))
cv2.waitKey()
