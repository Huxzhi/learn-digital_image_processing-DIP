import cv2
import numpy as np
img_path = '实验6-图像分割/numWrite.png'

img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)
# 边界方法需要先二值化
ret, thresh = cv2.threshold(imgGrey, 127, 255, cv2.THRESH_BINARY)


def 区域标志(img):
    h, w = img.shape
    dst = np.zeros((h, w))
    # flag = np.zeros(255)
    flag_temp = 1

    for x in range(1, h-1):
        for y in range(1, w-1):

            if flag_temp >= 255:
                print("flag over")
                return dst

            if img[x, y] == 0:

                dst[x, y] = flag_temp
                # 同步标号 左上、正上、右上及左前点
                if dst[x-1, y+1] != 0:
                    dst[x, y] = dst[x-1, y+1]
                elif dst[x-1, y] != 0:
                    dst[x, y] = dst[x-1, y]
                elif dst[x-1, y-1] != 0:
                    dst[x, y] = dst[x-1, y-1]
                elif dst[x, y-1] != 0:
                    dst[x, y] = dst[x, y-1]

                else:
                    # 对可能为 新物体进行标记
                    flag_temp += 1

                # 其中特别调整：当前点的右上点及左前点为不同标记，正上点 **或** 左上点不为物体，则当前点标记的左前同右上点置相同的值。
                if dst[x-1, y+1] != dst[x, y-1] and dst[x-1, y+1] and dst[x, y-1] and (img[x-1, y] == 255 or img[x-1, y-1] == 255):
                    dst[x, y] = dst[x-1, y+1]

                    for i in range(h):
                        for j in range(w):
                            if dst[i, j] == dst[x, y-1]:
                                dst[i, j] = dst[x-1, y+1]

                # print(flag)
    print(flag_temp)
    return np.uint8(dst)


'''
测量 周长的方法，就是统计边界的像素点，就是前面边界方法的 边界跟踪法 的改版，定义一个临时变量，最后/2 就行 ，我感觉可以进一步，斜对角长度记 根号2
测量 面积的方法，也是前面的 边界方法的 轮廓提取法，变量满足条件+1就行

'''


cv2.imshow("imgGrey", thresh)
区域 = 区域标志(thresh)
cv2.imshow("imgCeLiang", 区域)
# 本来是要对应像素值映射的，为了方便就 均衡化，也能看出不同区域不用颜色
cv2.imshow("imgCeLiang2", cv2.equalizeHist(区域))
cv2.waitKey()
