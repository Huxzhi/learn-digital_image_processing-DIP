"""
最重要的两种，邻域平均法 和 中值滤波

"""
import cv2
import numpy as np


img_path = "lena.png"


# 生成图片
img = cv2.imread(img_path)  # Mat 对象，和 numpy 里是一致的
# 生成灰色图片
imgGrey = cv2.imread(img_path, 0)

# define a threshold, 128 is the middle of black and white in grey scale
thresh = 128

# assign blue channel to zeros
img_binary = cv2.threshold(imgGrey, thresh, 255, cv2.THRESH_BINARY)[1]


# 后续看到了更好的写法 arr[x-a:x+a+1 ][y-a:y+a+1]
# 获得待处理像素点的邻域，简单的 n*n 矩阵
def kenerl(img, x: int, y: int, n: int):
    a = int((n-1)/2)
    arr = []

    for j in range(x-a, x+a+1):
        for k in range(y-a, y+a+1):
            arr.append(img[j][k])
    return arr


# n 是奇数
def 遍历(img, func, n: int):
    a = int((n - 1)/2)
    arr = np.copy(img)
    (w, h) = arr.shape
    for x in range(a, w-a):
        for y in range(a, h-a):
            arr[x][y] = func(kenerl(arr, x, y, n))
    return arr


def 邻域平均法(arr: list):
    return int(sum(arr)/len(arr))


def 中值滤波(arr: list):
    arr.sort()
    return arr[int((len(arr)-1)/2)]


# 复杂度太高了 11 就卡住了，还是小一点
img_ave = 遍历(img_binary, 邻域平均法, 5)
img_zzlb3 = 遍历(img_binary, 中值滤波, 3)
img_zzlb5 = 遍历(img_binary, 中值滤波, 5)
img_zzlb7 = 遍历(img_binary, 中值滤波, 7)

cv2.imshow("imgGrey", img_binary)
cv2.imshow("ave", img_ave)
cv2.imshow("zzlb3", img_zzlb3)
cv2.imshow("zzlb5", img_zzlb5)
cv2.imshow("zzlb7", img_zzlb7)

#  等待图片的关闭
cv2.waitKey()
