import cv2
import numpy as np


img_path = 'lena.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)

# 取中间正方形部分，也可以用半径表示


def 频率滤波器(img, 滤波器, name="1"):
    rows, cols = img.shape  # 图像的行(高度)/列(宽度)

    # 快速傅里叶变换(要对原始图像进行矩阵扩充)
    # rPad = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
    # cPad = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换
    imgEx = np.zeros((rows, cols, 2), np.float32)  # 对原始图像进行边缘扩充
    imgEx[:rows, :cols, 0] = img  # 边缘扩充，下侧和右侧补0
    dftImgEx = cv2.dft(imgEx, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
    dft_shift = np.fft.fftshift(dftImgEx)  # 变换到中间

    # 展示傅立叶变换后的图片
    dftImg = np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    dftImg = np.uint8(cv2.normalize(dftImg, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("imgDFT"+name, dftImg)

    # 实部和虚部都计算一遍
    dft_shift_pass = np.zeros(dft_shift.shape, dft_shift.dtype)
    for i in range(2):
        dft_shift_pass[:, :, i] = dft_shift[:, :, i] * 滤波器

    # 展示通过滤波后的图片
    dftImg_pass = np.log(
        1 + cv2.magnitude(dft_shift_pass[:, :, 0], dft_shift_pass[:, :, 1]))
    dftImg_pass = np.uint8(cv2.normalize(
        dftImg_pass, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("imgDFT_pass"+name, dftImg_pass)

    # 傅里叶逆变换
    idftImg = cv2.idft(np.fft.ifftshift(dft_shift_pass))  # 逆傅里叶变换
    idftMag = cv2.magnitude(idftImg[:, :, 0], idftImg[:, :, 1])  # 逆傅里叶变换幅值

    # 矩阵裁剪，得到恢复图像
    idftMagNorm = np.uint8(cv2.normalize(
        idftMag, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]

    return idftMagNorm


def 理想低通滤波器(img, a=40):

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)  # 计算频谱中心
    mask = np.zeros((rows, cols), dtype=np.float32)

    mask[crow - a: crow+a, ccol-a: ccol+a] = 1
    return mask


def 梯形低通滤波器(img, D0, D1):
    r, c = img.shape
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    low_pass = np.sqrt((u-r/2)**2 + (v-c/2)**2)

    idx = low_pass < D0
    idx2 = (low_pass >= D0) & (low_pass <= D1)
    idx3 = low_pass > D1

    low_pass[idx] = 1
    low_pass[idx2] = (low_pass[idx2] - D1) / (D0 - D1)
    low_pass[idx3] = 0

    return low_pass


def 巴特沃斯低通滤波器(img, D0, n):

    row, col = img.shape

    # 也可以用 双重for 循环写
    u, v = np.mgrid[0:row:1, 0:col:1]
    D = np.sqrt(np.power((u - row/2), 2) + np.power((v - col/2), 2))

    epsilon = 1e-8  # 防止除数为零
    # D(u,v) 是频率域的点(u,v) 到频率矩形中心的距离，D0是截止频率
    butterworth_low_pass = 1.0 / (1.0 + np.power(D / (D0 + epsilon), 2 * n))

    return butterworth_low_pass


def 高斯低通滤波器(img, D0):

    row, col = img.shape
    u, v = np.mgrid[0:row:1, 0:col:1]

    D = np.sqrt(np.power((u - row/2), 2) + np.power((v - col/2), 2))

    # D(u,v) 是频率域的点(u,v) 到频率矩形中心的距离，D0是截止频率
    Gaussian_low_pass = np.exp((-1)*D**2/2/(D0**2))

    return Gaussian_low_pass


cv2.imshow("imgGrey", imgGrey)
cv2.imshow("img_idea", 频率滤波器(imgGrey, 理想低通滤波器(imgGrey, 40), "idea"))
cv2.imshow("img_tixing", 频率滤波器(imgGrey, 梯形低通滤波器(imgGrey, 20, 100), "tixing"))
cv2.imshow("img_butterworth", 频率滤波器(
    imgGrey, 巴特沃斯低通滤波器(imgGrey, 20, 4), "butterworth"))
cv2.imshow("img_Gaussian", 频率滤波器(imgGrey, 高斯低通滤波器(imgGrey, 20), "Gaussian"))


# H_{hp}(u,v)=1-H_{lp}(u,v) 高通 = 1 - 低通
cv2.imshow("img_high_idea", 频率滤波器(
    imgGrey, 1 - 理想低通滤波器(imgGrey, 40), "high_idea"))

# 高频提升滤波
cv2.imshow("img_hb_idea", 频率滤波器(
    imgGrey, 2 - 理想低通滤波器(imgGrey, 40), "hb_idea"))


cv2.waitKey()
