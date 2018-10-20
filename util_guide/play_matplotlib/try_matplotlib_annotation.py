import matplotlib.pyplot as plt
import numpy as np

# 本节主要给 点做标识, 加上注解

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure(num=1, figsize=(8, 5))

# 设置取值范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))

plt.xlabel("i am x")
plt.ylabel("i am y")

# 这里要加 逗号,否则 plt.legend(handles=[l1, l2], loc='best') 中会报错
# plot 是画线, scatter 画点
l1, = plt.scatter(x, y2, label='up')
l2, = plt.plot(x, y1, label='down', color="red", linewidth=1.0, linestyle="--")

# handles=,labels=,
plt.legend(handles=[l1, l2], labels=['aaa', 'bbb'], loc='best')

# 标注点
x0 = 1
y0 = 2*x0 +1

plt.scatter()

plt.show()
