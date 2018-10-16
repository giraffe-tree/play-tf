import numpy as np

x1 = np.array([1,2])
x2 = np.zeros((1,2))

print(x1,x1.shape)
print(x2,x2.shape)

x3 = np.array([[1,2],[3,4],[5,6]])
print(x3[1:,1:],type(x3[1:,1:]),x3[1:,1:].shape)
print(x3[1:,:])

print(" - -- - - - -")
a = np.array([[1,2], [3, 4], [5, 6]])

print(a)
# 这里打印了 a[0,0],a[1,1]
print(a[[0, 1], [0, 1]])  # Prints "[1 4]"
print(a[0])
print(a[[0,1]])


