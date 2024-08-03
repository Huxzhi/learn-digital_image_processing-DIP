import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


img_path = "lena.png"
img = cv2.imread(img_path)

# gray = 0.11R + 0.59G + 0.3B
con = [0.11, 0.59, 0.3]

# 有好几种方式，还有，只取绿色，平均值等等


def 彩色变灰色(img, con):
    con.reverse()
    (h, w, s) = img.shape
    print(img.shape)
    # 小坑，数组是引用，需要用复制方法制造二维数组，不然每一行都一样
    arr = np.array([[0]*w]*h).astype("uint8")
    print(arr)
    for i in range(w):
        for j in range(h):
            for k in range(s):
                arr[i][j] += con[k] * img[i][j][k]

    return arr


imgGrey2 = 彩色变灰色(img, con)
print(imgGrey2)
cv2.imshow("grey", imgGrey2)


# 偷懒了，生成灰色图片,一个二维矩阵
imgGrey = cv2.imread(img_path, 0)
#  展示原图
cv2.imshow("img", imgGrey)


def 二值化(img, thresh):
    arr = np.copy(img)
    (h, w) = img.shape

    for x in range(w):
        for y in range(h):
            arr[x][y] = 0 if img[x][y] < thresh else 255
    return arr


def 灰度反转(img):

    arr = np.copy(img)
    (h, w) = img.shape

    for x in range(w):
        for y in range(h):
            arr[x][y] = 255 - img[x][y]
    return arr


def 直方图均衡化(img):
    (h, w) = img.shape
    arr = [0]*256
    for x in img.ravel():
        arr[x] += 1

    # 前缀和
    for index in range(len(arr) - 1):
        arr[index+1] += arr[index]

    for x in range(len(arr)):
        arr[x] = int(arr[x]*256 / (h*w) + 0.5)

    return np.array(arr).astype("uint8")


cv2.imshow("conver", 灰度反转(imgGrey))
cv2.imshow("b", 二值化(imgGrey, 150))

a = 直方图均衡化(imgGrey)


(h, w) = imgGrey.shape
out_img = np.copy(imgGrey)

for m in range(0, h):
    for n in range(0, w):
        out_img[m][n] = a[imgGrey[m][n]]

cv2.imshow("no", out_img)


dst = cv2.equalizeHist(imgGrey)
# 对比官方的，黑点更多，更加不平滑，哈哈，具体用 plt.show()可以展示分析一下
cv2.imshow("dst", dst)


# ravel() 把多维数组降为一维
plt.hist(out_img.ravel(), bins=256)
# 显示横轴标签
plt.xlabel("hex")
# 显示纵轴标签
plt.ylabel("count")
# 显示图标题
plt.title(" ")
# plt.show()


#  等待图片的关闭
cv2.waitKey()
