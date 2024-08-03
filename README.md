本人目前研 0，本科是软件工程，导师推荐的学习视频

## 学习资料

- 视频：Bilibili 数字图像处理-Digital Image Processing (DIP) https://www.bilibili.com/video/BV1tx41147Tx/
  - ppt 资料，链接： https://pan.baidu.com/s/12oO4WkX4KVwMqAAPrEK-2g 提取码：0f0o
- 【2022B 站最好的 OpenCV 课程推荐】OpenCV 从入门到实战 全套课程（附带课程课件资料+课件笔记）图像处理|深度学习人工智能计算机视觉 python+AI https://www.bilibili.com/video/BV1PV411774y/
  - up 主提供了，源代码，加他 vx 获取，写几个我认为关键的方法，熟悉一下

## 代码 code

因为视频是 2012 年的，所以使用 vc++ 6.0 ，对于现在的学习有些不必要，于是我用 python3 的 openCV 和 numPy 对 视频内出现的例子进行实现 ，也方便后续学习

大部分代码比较简单，有两个例子提一下

- 在 `实验6-图像分割/图像的测量.py` 区域标志中，我进行了重构，视频介绍用 c++实现的，是在难以理解，我用自己的理解方式写了一遍
- 在 `实验7-图像形态学/细化.py`，4 个判断条件比较奇怪，因为这是为了加快运行速度，减少迭代次数。其源头是 击中击不中的 8 个模板
- 一些语法糖，用`[i-1:i+2 , j-1:j+2]` 可以获取 `[i,j]` 为中心，`3*3`的矩阵
  - `imshow` 不能有中文，python3 用 utf-8 编码 是可以取中文方法名的

## 文档 docs

用 obsidian 做的笔记，都是课堂截图，用 批量替换的方式，把 `![[图像]]` 变成 `![](图像)` 忘记限定范围了，把代码也改了，我尽量改回来，有问题的地方可以提 issues

- 1-概述 计算机是怎么存储和计算 图像
  - 1a-图像存储格式 现在不用 bmp 存储，用 `cv2.imread()` 就灰色图像就是一个二维矩阵，彩色图像则是三维矩阵
- 2-图像编程处理 不用 c 语言，我用 openCV 的 python 实现
- 3-图像的几何变换 引入了齐次坐标，用 n+1 维矩阵来统一 n 维图形的变换操作，是将当前像素的坐标值，组合为一个向量，左乘转移矩阵后，得到变换后的坐标向量
  - 齐次坐标的**几何意义**相当于点（x，y）**投影**在 xyz **三维立体空间**的 z=1 的平面上。
- 4-图像增强 都是对单个像素进行处理，最重要的是**均衡化直方图**，使像素的灰度分布更均匀，我自己写的好像不太好，还好有现成的
- 5-图像的平滑处理 从邻域的角度出发，提出一个叫卷积核的模板，依次遍历图像的每个角落。比较有特点的是中值滤波，对模板内的像素点进行排列，取中间的值。由此延伸的一些特性，类似中位数的特点，比平均值更好的去除异常点
  - 这里中值滤波的范围影响很大，我只放大一点点参数，细节丢失很多，但是边缘还在
- 6-图像锐化处理及边缘检测 比较好理解，跟之前的学过的一样，有几大类别的算子，一阶微分，二阶微分；求边缘，去噪声
  - 6a-常用的微分算子 自己代码实现就 robert 和 Laplacian 像一点，其他几个都不像，我感觉 cv2 的实现，不是简单的卷积，有另外的判断，比如阈值
- 7-图像分割及测量 是图像处理的中间阶段，根据**不连续性**和**相似性** ，一般用灰度图或二值图
  - 灰度图根据阈值，提取目标物体和背景，变成二值图像
  - 二值图像再提取目标物体的轮廓
    - 轮廓提取法，掏空内部的点，四周都是目标物体，就删除，留下轮廓
    - 边界跟踪法：从一个边界点出发，一直意图右转，不行就直行，还不行就左转，肯定能回到原点，相当于绕最外围一圈
    - 区域增长法：根据某种相似性判断，进行合并和分裂，利用空间性质，但是开销比较大
    - 区域标记： 相邻像素标记为同一个物体 [图像分割及测量#^area-flag](./docs/7-图像分割及测量#^area-flag)
  - 测量：统计像素点，即可
    - 投影，分割的重要依据，但是没有细讲
    - 纹理分析，纹理是物体本身属性的反映，用于区分两种物体，其中一种方式是
      - 直方图统计特征分析法，计算相邻像素的均值和方差，叫 KS 检测方法
      - 边缘方向直方图分析法，统计水平和垂直方向上不相似（边界）个数
      - 图像自相关函数分析法，调整参数 d，观察是否有周期性，随着 d 增大，p (x, y) 下降较慢，则纹理较粗，反之，纹理较细，d 继续增加，会体现某种周期性
      - 灰度共生矩阵特征分析法，灰度共生矩阵被定义为从灰度为 i 的像素点出发，离开某个固定位置（相隔距离为 d，方位为）的点上灰度值为的概率，即，所有估计的值可以表示成一个矩阵的形式，以此被称为灰度共生矩阵。**对于纹理变化缓慢的图像，其灰度共生矩阵对角线上的数值较大；而对于纹理变化较快的图像，其灰度共生矩阵对角线上的数值较小，对角线两侧的值较大**。
- 8-图像的形态处理学 之前学的 [[数字图像处理) 就讲的比较好，多了一个**细化**，其源头是击中击不中的 8 个模板，需要多次迭代，把图像细化成骨架，在编程时，奇怪的条件，是为了加速计算，仔细观察两者条件是等价的
- 9-图像的变换域处理及应用 经典的傅立叶变换，对傅立叶函数不太懂，但是要明白在频域滤波的效果，理想滤波有振铃效果，高斯滤波较好，但是计算量大，巴特沃思滤波器计算快，且振铃效果不明显
  - 空间域的卷积和频率域的乘积效果近似
- 10-图像合成 简单的对应像素操作
- 11-彩色图像处理 增加了几个图像特效，比如，扩散，曝光，马赛克的实现
- 12-图像的小波变换处理 频率域里还待研究，空间域里，我已经完全理解了，奇数记录原始值，偶数记录差值，无损记录数据量没少，也可以放弃记录差值，则是分辨率降低一半
- 13-JPEG 图像的压缩编码 由多种压缩组合而成，傅立叶变换忽略细节，游程编码，霍夫曼编码，提高压缩比
- 14-图像处理在人脸识别的应用 了解一下实现过程，具体写，难度有点大，阈值判断条件不好定
