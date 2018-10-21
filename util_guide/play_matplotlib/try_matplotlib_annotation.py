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
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, label='down', color="red", linewidth=1.0, linestyle="--")

# handles=,labels=,
plt.legend(handles=[l1, l2], labels=['aaa', 'bbb'], loc='best')

# 标注点
x0 = 1
y0 = 1
# blue
plt.scatter(x0, y0, s=50, color='b')
# black
plt.plot([x0, x0], [y0, -2], 'k--', lw=2.5)

# method 1 加点的字注解
# 正则
plt.annotate(r'$x+y=%s$'%y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
# method 2
# plt.text(-1,1,r'$this\ us\ the\ some\ text.\ \mu\sigma_i\ \alpha_T$',
#          fontdict={'size':16,'color':'r'})

plt.show()




