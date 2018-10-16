import numpy as np

# 函数: split,array_split,vsplit,hsplit

A = np.arange(12).reshape((3,4))
print(A)

A1 = np.split(A,2,axis=1)
print(A1[0])
print(A1[1])

A2 = np.split(A,3,axis=0)
print(A2[0])
print(A2[1])

# 不等分隔
print(" \n  - - - \n")
A3 = np.array_split(A,3,axis=1)

print(A3[0])
print(A3[1])

# 其他函数

print(np.vsplit(A,3))
print(" - - ")
print(np.hsplit(A,2))