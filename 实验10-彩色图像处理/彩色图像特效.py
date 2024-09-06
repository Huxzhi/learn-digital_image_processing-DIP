import random
import cv2
import numpy as np


img_path = 'lena.png'
img = cv2.imread(img_path)

# print(img)


def 曝光处理(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    for i in range(h):
        for j in range(w):
            for k in range(s):
                if img[i][j][k] < 128:
                    dst[i][j][k] = img[i][j][k]
                else:
                    dst[i][j][k] = 255 - img[i][j][k]
    return np.uint8(dst)


def 扩散处理(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    for i in range(2, h-2):
        for j in range(2, w-2):
            dst[i, j] = img[i +
                            random.randint(-2, 2), j + random.randint(-2, 2)]
    return np.uint8(dst)


def 马赛克处理(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    print(h, w, s)
    for i in range(2, h-2, 5):
        for j in range(2, w-2, 5):
            for k in range(s):
                dst[i-2:i+3, j-2:j+3,
                    k] = np.full((5, 5), int(sum(map(sum, img[i-2:i+3, j-2:j+3, k])) / 25 + 0.5), dtype=np.uint8)

    return np.uint8(dst)


def 马赛克处理3(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    print(h, w, s)
    for i in range(1, h-1, 3):
        for j in range(1, w-1, 3):
            for k in range(s):
                dst[i-1:i+2, j-1:j+2,
                    k] = np.full((3, 3), int(sum(map(sum, img[i-1:i+2, j-1:j+2, k])) / 9 + 0.5), dtype=np.uint8)

    return np.uint8(dst)


def 马赛克处理2(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    for i in range(2, h-2, 5):
        for j in range(2, w-2, 5):
            for k in range(s):
                dst[i-2:i+3, j-2:j+3,
                    k] = np.full((5, 5), img[i, j, k], dtype=np.uint8)

    return np.uint8(dst)


def 浮雕处理(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    for i in range(1, h):
        for j in range(1, w):
            for k in range(s):
                dst[i, j, k] = img[i, j, k] - img[i-1, j, k]+128

    return np.uint8(dst)


def 霓虹处理(img):
    dst = np.zeros(img.shape)
    h, w, s = img.shape
    for i in range(h-1):
        for j in range(w-1):
            for k in range(s):
                dst[i, j, k] = 2 * np.sqrt((img[i, j, k] - img[i+1, j, k])
                                           ** 2 + (img[i, j, k] - img[i, j+1, k]) ** 2)

    return np.uint8(dst)


cv2.imshow("imgGrey", img)
# cv2.imshow("imgGreybaoguang", 曝光处理(img))
# cv2.imshow("imgGreykuosan", 扩散处理(img))
cv2.imshow("imgGreymasaike", 马赛克处理(img))
cv2.imshow("imgGreymasaike2", 马赛克处理2(img))
cv2.imshow("imgGreymasaike3", 马赛克处理3(img))
# cv2.imshow("imgGreyfudiao", 浮雕处理(img))
# cv2.imshow("imgGreynihong", 霓虹处理(img))

cv2.waitKey()
