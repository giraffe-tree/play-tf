import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,50)
# y=2*x+1
y = x*x

print(type(y))
plt.figure(num=1,figsize=(8,5))
plt.plot(x,y)

y2 = x*x*x
plt.plot(x,y2,color='red',linewidth=0.5,linestyle='--')

# 设置取值范围
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.xlabel('x 轴')
plt.ylabel('y 轴')

# 设置坐标轴上的参数 
new_ticks = np.linspace(-1,1,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3,],
			[r'$bad$',r'$bad2$',r'$\alpha$',r'$good$',r'$good2$'])

# 设置坐标轴的位置
# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))



plt.show()


