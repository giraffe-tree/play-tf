import matplotlib.pyplot as plt
import numpy as np

# 本节主要给 点做标识, 加上注解

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()

# 设置取值范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))

plt.xlabel("i am x")
plt.ylabel("i am y")


# 这里要加 逗号,否则 plt.legend(handles=[l1, l2], loc='best') 中会报错
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, label='down', color="red", linewidth=1.0, linestyle="--")

# handles=,labels=,
plt.legend(handles=[l1, l2], labels=['aaa', 'bbb'], loc='best')



plt.show()
