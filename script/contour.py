# 导入模块
import numpy as np
import matplotlib.pyplot as plt

# 建立步长为0.01，即每隔0.01取一个点
step = 0.1
x = np.arange(-101, 101, step)
y = np.arange(-101, 101, step)
# 也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份

# 将原始数据变成网格数据形式
X, Y = np.meshgrid(x, y)
# 写入函数，z是大写
Z1 = np.sqrt((X + 60) ** 2 + (Y + 20) ** 2)
Z2 = np.sqrt((X + 20) ** 2 + (Y + 60) ** 2)
Z3 = np.sqrt((X - 40) ** 2 + (Y - 40) ** 2)
# 画等高线
plt.contour(X, Y, Z1, 8, colors='#0a5f38', linestyles='dashed', alpha=0.6)
plt.contour(X, Y, Z2, 8, colors='#20c073', linestyles='dashed', alpha=0.6)
plt.contour(X, Y, Z3, 8, colors='#fc5a50', linestyles='dashed', alpha=0.6)
plt.gca().set_aspect(1)
plt.show()
