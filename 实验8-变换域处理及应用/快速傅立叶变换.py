import cv2
import numpy as np


img_path = 'lena.png'
img = cv2.imread(img_path)
imgGrey = cv2.imread(img_path, 0)


def 快速傅立叶变换(img):
    rows, cols = img.shape  # 图像的行(高度)/列(宽度)

    # 快速傅里叶变换(要对原始图像进行矩阵扩充)
    rPad = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
    cPad = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换
    imgEx = np.zeros((rPad, cPad, 2), np.float32)  # 对原始图像进行边缘扩充
    imgEx[:rows, :cols, 0] = img  # 边缘扩充，下侧和右侧补0
    dftImgEx = cv2.dft(imgEx, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
    dft_shift = np.fft.fftshift(dftImgEx)  # 变换到中间
    dftImg = np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dftImgEx[:, :, 1]))
    dftImg = np.uint8(cv2.normalize(
        dftImg, None, 0, 255, cv2.NORM_MINMAX))

    # 傅里叶逆变换
    idftImg = cv2.idft(dftImgEx)  # 逆傅里叶变换
    idftMag = cv2.magnitude(idftImg[:, :, 0], idftImg[:, :, 1])  # 逆傅里叶变换幅值

    # 矩阵裁剪，得到恢复图像
    idftMagNorm = np.uint8(cv2.normalize(
        idftMag, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    imgRebuild = np.copy(idftMagNorm[:rows, :cols])

    return [dftImg, imgRebuild]


cv2.imshow("imgGrey", imgGrey)
dft, imgRebuild = 快速傅立叶变换(imgGrey)
cv2.imshow("imgDFT", dft)
cv2.imshow("imgreDFT", imgRebuild)
cv2.waitKey()
