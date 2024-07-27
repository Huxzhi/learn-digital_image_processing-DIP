import cv2
import numpy as np


img_path = "cell.png"


# 生成灰色图片
imgGrey = cv2.imread(img_path, 0)


def 迭代阈值分割(img):
    T1 = T2 = 127

    h, w = img.shape
    arr = [0]*256
    for x in img.ravel():
        arr[x] += 1

    temp1 = temp2 = 0
    while (True):
        #  求平均灰度值
        temp1 = sum(i*j for i, j in enumerate(arr[:T1])) / sum(arr[:T1])
        temp2 = sum(i*j for i, j in enumerate(arr[T1:])) / sum(arr[T1:])
        T2 = int((temp1 + temp2)/2)

        # 总是波动，不能相等
        if abs(T1 - T2) < 2:
            break
        else:
            print(T2)
            T1 = T2
    return T1


阈值 = 迭代阈值分割(imgGrey)

ret, thresh = cv2.threshold(imgGrey, 阈值, 255, cv2.THRESH_BINARY)


#  展示灰色图片
cv2.imshow("imgGrey", imgGrey)
cv2.imshow("thresh", thresh)
#  等待图片的关闭
cv2.waitKey()
