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

plt.show()



