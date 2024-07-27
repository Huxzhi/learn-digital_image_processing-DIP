import cv2
import numpy as np


def 马赛克显示(img, x=0, y=0, w=300, h=300, size=12):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param img: opencv img
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int size:  马赛克每一块的宽

    用了 cv2.rectangle 的填充方法
    """
    arr = np.copy(img)  # 返回一个复制的图像
    ih, iw = arr.shape[0], arr.shape[1]

    for i in range(0, min(y+h, ih - size), size):  # 关键点0  防止溢出
        for j in range(0, min(x+w, iw - size), size):
            color = arr[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (j + x, i + y)
            right_down = (j + x + size - 1, i + y + size - 1)  # 关键点2 减去一个像素
            cv2.rectangle(arr, left_up, right_down, color, -1)
    return arr


img_path = "lena.png"


# 生成图片
img = cv2.imread(img_path)  # Mat 对象，和 numpy 里是一致的
马赛克 = 马赛克显示(img, 100, 100, 100, 100, 10)

#  展示原图
cv2.imshow("img", img)
#  展示灰色图片
cv2.imshow("mo", 马赛克)


#  等待图片的关闭
cv2.waitKey()
