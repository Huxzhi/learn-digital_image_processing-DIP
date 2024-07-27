import cv2
import numpy as np

img_path = '实验5-边缘检测/cell.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)

ret, thresh = cv2.threshold(imgGrey, 70, 255, cv2.THRESH_BINARY)


# https://sci-hub.usualwant.com/10.1145/357994.358023
# https://www.cnblogs.com/xiaxuexiaoab/p/15842240.html
def 细化(img):
    h, w = img.shape
    dst = np.copy(img)

    p = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1],
                 [1, 0], [1, -1], [0, -1], [-1, -1]])
    flag = True
    flag_count = 0
    while (flag):
        flag = False
        # 第一步遍历
        for x in range(2, h-2):
            for y in range(2, w-2):
                if dst[x, y] > 127:
                    continue
                # 视频里 进行反转 把目标点变1，非目标点为0，方便进行与运算
                s = dst[x-2:x+3, y-2:y+3].copy()
                s[s < 1] = 1
                s[s > 2] = 0

                # 条件1
                num = sum(s[1:4, 1:4].ravel())
                if num < 2 or num > 6:
                    continue

                # 条件2:计算 P(A)
                PA = 0
                for i in range(8):
                    if s[2 + p[(i+1) % 8][0], 2 + p[(i+1) % 8][1]] \
                            - s[2 + p[i][0], 2 + p[i][1]] == 1:
                        PA += 1
                if PA != 1:
                    continue

                # 顺时针转
                # p9 p2 p3
                # p8 p1 p4
                # p7 p6 p5

                # 条件3 p2*p4*p8 = 0 同时 P(s[1][2])!=1
                if s[1, 2] * s[2, 1] * s[2, 3] != 0:

                    PA = 0  # 以 s[1][2] 为中点
                    for i in range(8):
                        PA += 1 if s[1 + p[(i+1) % 8][0], 2 + p[(i+1) % 8][1]] \
                            - s[1 + p[i][0], 2 + p[i][1]] == 1 else 0
                    if PA == 1:
                        continue

                # 条件4 p2*p6*p8 = 0 同时 P(s[2][1])!=1

                if s[1, 2] * s[2, 1] * s[3, 2] != 0:

                    PA = 0  # 以 s[2][1] 为中点
                    for i in range(8):
                        PA += 1 if s[2 + p[(i+1) % 8][0], 1 + p[(i+1) % 8][1]] \
                            - s[2 + p[i][0], 1 + p[i][1]] == 1 else 0
                    if PA == 1:
                        continue

                # 4个条件都不满足 ，则删除
                dst[x, y] = 255
                # print((x, y))
                flag = True
                flag_count += 1
    print("迭代次数：", flag_count)
    return dst


def 粗化(img):
    dst = 255 - img
    return 细化(dst)


cv2.imshow("imgGrey", thresh)
cv2.imshow("imgXiHua", 细化(thresh))
cv2.imshow("imgCuHua", 粗化(thresh))
cv2.waitKey()
