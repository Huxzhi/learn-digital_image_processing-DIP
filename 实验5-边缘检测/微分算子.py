
import math
import cv2
import numpy as np


img_path = "实验5-边缘检测/cell.png"


# 生成灰色图片
imgGrey = cv2.imread(img_path, 0)


def roberts(img):
    h, w = img.shape
    new_image = np.zeros((h, w))

    for i in range(h-1):
        for j in range(w-1):
            # abs(f[i,j] - f[i+1,j+1] ) + abs( f[i+1 ,j] - f[i,j+1] ) # 绝对值相加，也跟事例不同
            new_image[i, j] = math.sqrt(pow(img[i, j] - img[i+1, j+1], 2) +
                                        pow(img[i+1, j] - img[i, j+1], 2))
    return np.uint8(new_image)


def sobel(img):
    h, w = img.shape
    new_image = np.zeros((h, w))

    Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for i in range(1, h-1):
        for j in range(1, w-1):
            new_image[i, j] = max(
                abs(np.sum(img[i-1:i+2, j-1:j+2] * Gx)), abs(np.sum(img[i-1:i+2, j-1:j+2] * Gy)))
    return np.uint8(new_image)


def prewitt(img):
    h, w = img.shape
    new_image = np.zeros((h, w))

    Gx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Gy = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    for i in range(1, h-1):
        for j in range(1, w-1):
            new_image[i, j] = abs(
                np.sum(img[i-1:i+2, j-1:j+2] * Gx)) + abs(np.sum(img[i-1:i+2, j-1:j+2] * Gy))
    return np.uint8(new_image)


# 8个模板，可以用for生成，竟然没有短
def Krisch_kernel():
    kernel = np.zeros((8, 3, 3))
    arr = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
    for i in range(8):
        for j in range(3):
            for k in range(3):
                if j == 1 and k == 1:
                    continue
                kernel[i][j][k] = 5 if ((arr[k][j]+i) % 8) > 4 else -3
    return kernel


def Krisch(img):
    h, w = img.shape
    new_image = np.zeros((h, w))
    # 有8个模版，在3x3 的模板中，3个连续上下或左右的 5 ，表示边缘的方向
    kernel = Krisch_kernel()
    print(kernel)
    for i in range(1, h-1):
        for j in range(1, w-1):
            temp = 0
            for k in range(len(kernel)):
                new_image[i, j] = max(
                    abs(np.sum(img[i-1:i+2, j-1:j+2] * kernel[k])), temp)
    return np.uint8(new_image)


def laplace(img):
    h, w = img.shape
    new_image = np.zeros((h, w))
    # operator = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0))
    operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    for i in range(1, h-1):
        for j in range(1, w-1):
            new_image[i, j] = abs(np.sum(img[i-1:i+2, j-1:j+2] * operator))
    return np.uint8(new_image)


def LOG(img):

    h, w = img.shape
    new_image = np.zeros((h, w))

    operator = np.array([[-2, -4, -4, -4, -2],
                         [-4, 0, 8, 0, -4],
                         [-4, 8, 24, 8, -4],
                         [-4, 0, 8, 0, -4],
                         [-2, -4, -4, -4, -2]])

    for i in range(2, h-2):
        for j in range(2, w-2):
            new_image[i, j] = np.sum(img[i-2:i+3, j-2:j+3] * operator)
    return np.uint8(new_image)


#  展示灰色图片
cv2.imshow("imgGrey", imgGrey)


cv2.imshow("roberts", roberts(imgGrey))  # 怎么效果都不好


cv2.imshow("sobel", sobel(imgGrey))

grad_X = cv2.Sobel(imgGrey, cv2.CV_64F, 1, 0)
grad_Y = cv2.Sobel(imgGrey, cv2.CV_64F, 0, 1)
sobelx = cv2.convertScaleAbs(grad_X)   # 转回uint8
sobely = cv2.convertScaleAbs(grad_Y)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imshow("cv2sobel", sobel(sobelxy))

cv2.imshow("prewitt", prewitt(imgGrey))


cv2.imshow("Krisch", Krisch(imgGrey))


cv2.imshow("laplace", laplace(imgGrey))

gray_lap = cv2.Laplacian(imgGrey, cv2.CV_16S, ksize=3)
# 转回uint8
cv2.imshow("cv2lap", cv2.convertScaleAbs(gray_lap))


# 先通过高斯滤波降噪
gaussian = cv2.GaussianBlur(imgGrey, (3, 3), 0)

# 再通过拉普拉斯算子做边缘检测
dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
imgLOG = cv2.convertScaleAbs(dst)

cv2.imshow("LOG", LOG(imgGrey))
cv2.imshow("cv2LOG", imgLOG)

#  等待图片的关闭
cv2.waitKey()
