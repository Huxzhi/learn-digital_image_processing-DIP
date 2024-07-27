"""
将气球画在图的上部四分一处，但图的大小为1，显示时将图像的上部四分一处复制四次，分别显示在右图。
图像中的红气球物体左右、上下、对称复制功能。

"""

import numpy as np
import cv2


img_path = "实验一/红气球图片.png"

img = cv2.imread(img_path)

img2 = img.copy()
arr = img2[0:100, 0:100]
print(arr)
arr2 = cv2.flip(arr, 1)
arr3 = cv2.flip(arr, 0)
arr4 = cv2.flip(arr, -1)
harr = cv2.hconcat([arr, arr2])
harr2 = cv2.hconcat([arr3, arr4])
img2 = cv2.vconcat([harr, harr2])

cv2.imshow("img", cv2.hconcat([img, img2]))

#  等待图片的关闭
cv2.waitKey()
