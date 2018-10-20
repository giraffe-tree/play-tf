import matplotlib.pyplot as plt
import numpy as np

# 本节主要给 线做标识, 这个线是什么

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()

plt.xlim((-1, 2))
plt.ylim((-2, 3))

plt.xlabel("i am x")
plt.ylabel("i am y")

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.5, -1, 1.0, 3, ],
           [r'$really bad$', r'$bad$', r'$normal$', r'$good$', r'$really good$'])

# 这里要加 逗号,否则 plt.legend(handles=[l1, l2], loc='best') 中会报错
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, label='down', color="red", linewidth=1.0, linestyle="--")

# handles=,labels=,
plt.legend(handles=[l1,l2],labels=['aaa','bbb'] ,loc='best')

plt.show()
