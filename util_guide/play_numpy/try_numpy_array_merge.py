import numpy as np

x1 = np.array([1,1,1])
x2 = np.array([2,2,2])

# 左右合并
x3 = np.hstack((x1,x2))
print(x3)

# 上下合并
x4 = np.vstack((x1,x2))
print(x4)

# 增加一个维度
print(x1[np.newaxis,:])

a1 = x1[:,np.newaxis]
a2 = x2[:,np.newaxis]

print(np.hstack((a1,a2)))

print(" - - -- - - ")
# 多维合并
a3 = np.concatenate((a1,a2,a2,a1),axis=1)
print(a3)

