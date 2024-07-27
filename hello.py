
import cv2
import numpy as np


img_path = "lena.png"


# 生成图片
img = cv2.imread(img_path)  # Mat 对象，和 numpy 里是一致的
# 生成灰色图片
imgGrey = cv2.imread(img_path, 0)
# img = 马赛克显示(img, 100, 100, 100, 100, 10)


print(img.shape)  # （高度，宽度，通道数）
print(img.size)  # 像素总数目
print(img.dtype)
# print(img)  # 3个二维矩阵，分别是 B, G, R


#  展示原图
cv2.imshow("img", img)
#  展示灰色图片
cv2.imshow("imgGrey", imgGrey)


#  等待图片的关闭
cv2.waitKey()
# 保存灰色图片
cv2.imwrite("Copy.jpg", imgGrey)
