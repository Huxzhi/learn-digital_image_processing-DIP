"""

当原图的像素点为蓝色(0,115,0)时，将显示屏幕上该位置的像素点像素值置为绿色(0,0,115); 否则不变，将原图像素值赋给新图。

"""
import numpy as np
import cv2


def 颜色替换(img, src_clr, dst_clr):
    img_arr = np.asarray(img, dtype=np.double)
    src_clr.reverse()  # rgb 2 bgr
    dst_clr.reverse()

    dst_arr = img_arr.copy()
    for i in range(img_arr.shape[1]):
        for j in range(img_arr.shape[0]):
            if (img_arr[j][i] == src_clr)[0] == True:
                dst_arr[j][i] = dst_clr

    return np.asarray(dst_arr, dtype=np.uint8)


img_path = "实验一/蓝色圆形.png"
# RGB 的顺序
org = [0, 0, 115]
rep = [0, 115, 0]


img = cv2.imread(img_path)

img2 = 颜色替换(img, org, rep)

cv2.imshow("img", np.hstack([img, img2]))

#  等待图片的关闭
cv2.waitKey()
