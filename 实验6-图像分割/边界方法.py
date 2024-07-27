import cv2
import numpy as np


img_path = '实验6-图像分割/cup.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)

# 边界方法需要先二值化
ret, thresh = cv2.threshold(imgGrey, 80, 255, cv2.THRESH_BINARY)


def 轮廓提取法(img):
    h, w = img.shape
    G = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    dst = np.copy(img)
    for x in range(1, h-1):
        for y in range(1, w-1):
            # 四周8个点都为黑，则赋 255
            if not (img[x-1:x+2, y-1:y+2] * G).any():
                dst[x, y] = 255
    return dst

# 从图像中一个边界点出发，然后根据某种判别准则搜索下一个边界点，以此跟踪出目标边界。


def 边界跟踪法(img):
    h, w = img.shape
    # 按 逆时针顺序，从左上角开始
    arr = np.array([[-1, 1], [0, 1], [1, 1], [1, 0],
                    [1, -1], [0, -1], [-1, -1], [-1, 0]])

    dst = np.full((h, w), 255, dtype=np.uint8)

    height = 'height'
    width = 'width'

    findStartPoint = False
    startPoint = {height: 0, width: 0}
    currentPoint = {height: 0, width: 0}
    prePoint = {height: 0, width: 0}
    # 从左下角开始
    for x in range(h-1, -1, -1):
        for y in range(w):
            # 找到一个起始点，且不是已标记的边界
            if img[x, y] == 0 and (not findStartPoint) and dst[x][y] == 255:
                findStartPoint = True
                startPoint[height] = x
                startPoint[width] = y
                dst[x][y] = 0
                print((x, y))

    # 开始沿着起始点
    beginDirect = 0
    findStartPoint = False
    currentPoint[height] = startPoint[height]
    currentPoint[width] = startPoint[width]
    while (not findStartPoint):
        findPoint = False
        while (not findPoint):

            # 探测点的值
            prePoint[height] = currentPoint[height] + arr[beginDirect][0]
            prePoint[width] = currentPoint[width] + arr[beginDirect][1]

            pixel = img[prePoint[height], prePoint[width]]

            if pixel == 0:
                findPoint = True
                print(pixel, prePoint)
                currentPoint[height] = prePoint[height]
                currentPoint[width] = prePoint[width]

                dst[currentPoint[height], currentPoint[width]] = 0

                beginDirect = (beginDirect-2 + 8) % 8

            # 思考半天，这里不能加 else
            beginDirect = (beginDirect+1) % 8
            # 绕了一圈 回到原点
            if prePoint[height] == startPoint[height] and prePoint[width] == startPoint[width]:
                findStartPoint = True
                findPoint = True

    return dst


"""
区域增长法：实现方式 和 二值化差不多，区别是阈值由鼠标选择，就不写了


区域分裂合并法：由四叉树实现，是游戏引擎为了优化性能的重要手段，没有统一的实现方式

"""


cv2.imshow("img", thresh)
cv2.imshow("imgLunKuo", 轮廓提取法(thresh))
cv2.imshow("imgbianjie", 边界跟踪法(thresh))
cv2.waitKey()
