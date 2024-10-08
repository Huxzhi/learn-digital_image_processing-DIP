---
created: 2024-07-26T19:05
updated: 2024-07-26T22:45
---

## 压缩编码基础

1. **压缩的原因**: 数字视频码率高达 216 Mb/s。数据量之大，无论是网络传输，还是存储都构成巨大压力。在保持信号质量的前提下，要降低码率及数据量。
2. **压缩的原理**: 图像信息存在着大量的规律性或相关性，在传输的前一个样值中包含了后一个样值或后一帧中相关位置的样值内容。

### 目标

① 去除信息中的相关性，去除冗余码，使样值独立，降低信息码流。
② 可以采用一些特殊的编码方式，使平均比特数降低，从而可进一步降低信息码流。
(4) 信源编码: 降低码率的过程，称为压缩编码，也叫信源编码。

### 方法

编码方式是多种多样的，不同的算法其压缩率也不同，但都应本着无损的原则。在实际应用中往往是采用多种不同算法的综合压缩编码方式，反复压缩，以取得较高的压缩率。

#### 压缩编码基础--莫尔斯码

电报码: 是采用“· ”和“—”来表示 26 个英文字母的变字长编码。
编码思想:

1. 常用字母用短码表示。
2. 不常用的字母用长码表示。

编码方法：通过变字长编码方式。对常用英文单词进行的大量统计。找出各字母出现的概率，最后确定:

12 个字母（出现几率最小）用 4 bit 数字表示；
8 个字母（出现几率较少的）用 3 bit 数字表示；
4 个字母（出现几率较高的）用 2 bit 数字表示；
2 个字母（出现几率最高的）用 1 bit 数字表示，
共 26 个字母。

每个字母的平均码长为: 平均码长=(48+24+8+2)÷26=3.15 bit/字母

要用固定码长方式则需要 25 =32, 即 5 bit 来表示。

#### 压缩编码基础—预测编码

**差值编码原理**

样值与前一个（相邻）样值的差值，则这些差值绝大多数是很小的或为零，可以用短码来表示，而对那些出现几率较少的较大差值，用长码来表示，则可使总体码数下降。

**采用对相邻样值差值进行变字长编码的方式称为差值编码，又称为差分脉码调制（DPCM）**
![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_19-31-53-457.png)

#### 霍夫曼 (Huffmun) 编码

（1）变字长编码：对信源中出现概率大的“对象”用短码表示，对出现概率较小的“对象”用长码表示。其可获得较
短的平均码长。
注：
“对象” 只是一个欲编码的数据、符号或元素。

① 将信源对象按出现的概率由大到小排序。
② 找出最小两个概率点，大为“1”，小为“0”，如概率相等，随意“0”和“1”分配。
③ 将这两个概率点的概率相加，生成一新的概率点。
④ 再在新生成概率点与余下概率点中再选出两个最小比较，大为“1”，小为“0”。
⑤ 再求和，生成一新的概率点，以此类推，直至新的概率点的概率为 1 为止。
⑥ 最后将对应各“对象”的数码，按结构顺序组合起来，即为各信源“对象”的霍夫曼编码。

> 408 学过，不会出现前缀相同的编码

### 压缩编码基础

有损压缩

#### 压缩编码基础 —变换编码

(1) **变换的原因**：信号的相关性不仅表现在位置空间（空域）中，在其他的域中也具有很强的相关性，因此压缩编码的方法并不唯一。
(2) **不同域有不同特点**：静止图像的位置相关性较强，运动图像的频率相关性较强，因此在空域中解决不了的问题在频域中就可以解决。

#### 压缩编码基础 —离散余弦变换（DCT）

(1) **图像的频率特征**：低频信号幅值大，高频信号幅值小。信号能量主要集中于低频分量, 而高频分量的能量较小。
(2) **相关性分析**：将信号变换到频率域中，幅值大的低频分量集中在一个区域，幅值小的高频分量分布在其他位置，表现出了较强的频率相关性。DCT 编码就是这种效率更高，便于压缩的编码方法。

① 分块：将每个分量图像分成许多 8×8=64 个样点组成的像块，得到在空域中的 8×8 的样值矩阵。
② 变换：利用 FDCT 公式，将空域中的 8×8 样值矩阵，正向变换为频域中的 8×8 DCT 系数矩阵。

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-07-17-986.png)

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-07-54-825.png)

F（0，0）对应直流分量，称为 DC 系数，其它 63 个对应交流分量的系数，称为 AC 系数。

两个空间的同位置系数无对应关系。

在频域中的右下角对应高频部分，而在左上角对应低频部分（特点，相关性）。

DC 系数为空域中 64 个样值的平均值。

**DCT 系数规律**：低频系数值大，高频系数值小。
对比两个数值矩阵观察相关性

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-11-05-343.png)

逆变换，是无损变换
![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-11-29-092.png)

#### 量化

量化的原因： DCT 之后其系数矩阵中相关性不够明显，为进一步降低 DCT 系数矩阵中非零系数的幅值，增加零系数的个数，使相关性表现的更明显，需要进一步量化。

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-12-48-649.png)

对 DCT 系数矩阵中的每一个值逐一量化。
F(U，V)为 DCT 系数矩阵中位于(U，V)的 DCT 系数;
W(U，V)为量化表中位于(U，V)点的量化步长，（不同位置可以采用不同的量化步长）;
Q(U，V)为对应于(U，V)位置的量化值。
round（）为取整函数。

① 对失真的要求：量化是图像质量下降的重要原因，DCT 系数量化是基于限失真编码理论进行的，**容许有失真**，但应在视觉容许的容限内。
② 视觉要求：
a. 对亮度信号与色度信号的分辨能力不同;
b. 对低频图像信号和高频图像信号的分辨能力不同。
结论：可以采用不同的量化方案。

**量化步长表**

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-14-22-203.png)

a.量化表的区域性 ： DCT 系数矩阵中不同的区域采用不同的量化步长 （高低频的区别）

b.量化表的多样性：对不同的分量信号采用不同的量化表（不同分量信号的区别）

**量化表的可变性**：是比较理想的，还可以改变。
区域滤波法：属于均匀量化方式（对每点而言）。

**信号数字化中的量化与 DCT 系数量化的区别**：前者为描述信号幅值，后者降低信号幅度。
逆量化 Q-1：接收端，一定要使用与发送端相同的量化表进行逆量化，方可使图像还原。

**视觉加权法**：采用统一的量化步长，再配以 8×8 视觉加权矩阵，其中对应于 DC 的权值最高为 1。而对应于 AC 的权值都小于 1，对应于高频的权值为最小。
视觉加权的量化方案采用下列公式：

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-17-49-759.png) 式中 W 为统一步长，K (U，V) 为加权系数。

#### 压缩编码基础—Zig-Zag 扫描

Zig-Zag 扫描：一种将二维数组转变为一维数组的 Z 字形扫描方法。

1. **采用扫描的原因**：量化后的 DCT 系数仍然是二维系数矩阵，无法直接传输，还需将其变为一维数据序列。对 Q 矩阵重新排列。
2. **Zig-Zag 扫描的依据**：在量化后的 DCT 系数矩阵中，非 0 的数据主要都集中于矩阵的左上角。
3. **Zig-Zag 扫描的方法**：Zig-Zag 扫描采用的是 Z 字形扫描方式。从直流分量 DC 开始进行 Z 字形扫描。
   1. ![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-19-27-533.png)
4. **Zig-Zag 扫描序列**：系数矩阵 Q，进行 Zig-Zag 扫描所得到的数据序列。
   1. ![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-20-49-830.png)
5. (5) **Z 扫描的特点**
   1. 可以增加连续 0 系数的个数，也就是增加 0 的游程长度。
   2. 在数据序列中，非零系数主要都集中于数据序列的首部，在数据序列的尾部，则都是连零 (EOB) 数据。这样对传输中的数据压缩十分有利。

#### 压缩编码基础 —游程编码 (RLC)

**游程编码 (RLC)**: 消去一维数组序列尾部连续 0 数据的编码方法。

1. 游程：连续 0 的长度，或连续 0 的个数。
2. 游程编码的方法：将一维数组序列转化为一个由二元数组 (run, level) 组成的数组序列。其中：
   1. ①run 表示连续 0 的长度；
   2. ②level 表示连续 0 之后的一个非零值；
   3. ③ 用 EOB 表示后面所有剩余的连续 0。
3. **游程编码实例**（10 进制）：对应以上的两个一维数组序列的游程编码为：
   1. ![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-23-40-399.png)
4. **16 进制的编码方法**：每两个字节为一个字符对。 1) ![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_20-25-25-292.png)
   注：① 第一字节中：高 4 位表示一维数组序列中非零系数前零的个数。低 4 位则表示这个非零系数所需的比特数。 （告诉非零系数的比特数 **推测**是加快解码速度）
   ② 第二字节：完全用于表示非零系数的数值。
   ③ EOB 用 FFFF 表示。
   ④ 负数在此用补码表示。

#### 压缩编码基础 —熵编码

**熵编码**：是一种可变字长编码。

(1) **游程编码后的熵编码**：在变换编码中，经过游程编码后的字符对数组序列，并不直接用于数据传输，还要对其进行霍夫曼编码，以进一步提高数据压缩率.
(2) **熵编码**：在发送端，根据字符对出现的概率进行霍夫曼编码，形成一个码表 (霍夫曼表) 存储在编码器的 ROM 中，传输时, 按码表把字符对“翻译”成对应的二进制数码 (霍夫曼码)。
(3) **熵解码**：在接收端, 则必须采用同样的霍夫曼码表解码。

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_21-12-52-883.png)

## JPEG 压缩

JPEG 压缩是一种针对静止的连续色调的图像压缩方法.

**JPEG 压缩编码-解压缩算法框图**

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_21-14-25-747.png)

1. **压缩比可控**: 编码器的参数中应包括控
   制压缩比和图像质量的成分。
2. **不受限制**: 适用于所有的连续色调图像，不应对图像的尺寸、彩色空间和像素纵横比等特性进行限制，也不应对图像的场景内容 (如复杂性、彩色范围或统计特性) 有任何限制。
3. 适中的计算复杂性: 压缩算法既可用软件实现，也可用硬件实现，并且具有较好的性能。
4. 具有下述 4 种操作模式：① 顺序编码 ② 累进编码 ③ 无失真编码 ④ 层次编码

### 基于 DCT 编码的 JPEG 压缩过程

![](./assets/image-13-JPEG 图像的压缩编码-2024-07-26_21-17-11-686.png)

后续是 jpeg 具体的实现就不写了
