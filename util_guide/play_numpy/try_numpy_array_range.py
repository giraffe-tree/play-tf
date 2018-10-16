import numpy as np

x = np.arange(3,15).reshape(3,4)
# print(x,x.size)
# print(x[1][1])
# print(x[1,1])

# print(x[[0,1,1],[0,0,1]])

# 迭代行
for row in x :
	print(row)

# 迭代列
for col in x.T:
	print(col)

# 转成一行
print(x.flatten())
for item in x.flat:
	print(item)
