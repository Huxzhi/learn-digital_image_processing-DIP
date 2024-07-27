import numpy as np

w, h = 3, 4  # 宽高

x1, y1 = 1, 2  # 平移距离

# 将创建一个用指定形状用 1 填充的数组。默认的dtype是float64
m = np.ones((w, h))  # 测试矩阵
res = np.zeros((w+x1, h+y1))  # 保存结果
print(m)


def 几何变换(转移方程):
    for x in range(w):
        for y in range(h):
            # 左乘 转移方程
            mx = 转移方程 @ np.array([[x], [y], [1]])
            # 比较简单，其他操作比如旋转，镜像，都只要改上面的转移方程就好了，还以为需要单独写
            res[mx[0][0]][mx[1][0]] = m[x][y]


平移 = [[1, 0, x1], [0, 1, y1], [0, 0, 1]]

fWidth = 100  # 总宽度
fHeight = 100  # 总高度
水平镜像 = [[-1, 0, fWidth], [0, 1, 0], [0, 0, 1]]
垂直镜像 = [[1, 0, 0], [0, -1, fHeight], [0, 0, 1]]


几何变换(平移)
print(res)
