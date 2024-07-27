import cv2
import numpy as np


img_path = 'lena.png'
img_path2 = '实验5-边缘检测/cell.png'

imgGrey = cv2.imread(img_path, 0)
imgGrey2 = cv2.imread(img_path2, 0)

ret, thresh = cv2.threshold(imgGrey, 128, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(imgGrey2, 128, 255, cv2.THRESH_BINARY)

rows, cols = imgGrey.shape

imgEx = np.zeros(imgGrey.shape)  # 对添加图像进行边缘扩充
rows2, cols2 = imgGrey2.shape
imgEx[:rows2, :cols2] = thresh2[:, :]  # 边缘扩充，下侧和右侧补0


u = np.arange(rows)
v = np.arange(cols)
u, v = np.meshgrid(u, v)
low_pass = np.sqrt((u-rows/2)**2 + (v-cols/2)**2)

r = 80  # 半径为 r 的圆

idx = low_pass < r
idx2 = low_pass >= r

low_pass[idx] = 1
low_pass[idx2] = 0


cv2.imshow("imgGreyjia", thresh / 2 + imgEx/2)
cv2.imshow("imgGreycheng", thresh * low_pass)
cv2.waitKey()
