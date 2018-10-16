import numpy as np

a = np.arange(10)
b = a
a[0]=11
print(b)
# return True
print(a is b)
b[1] = 12
print(a)
a[2:4]=[9,10]
print(b)

# deep copy
print("deep copy begin...")
x = a.copy()
print(x)
x[1] = 1000
# a 不会受到影响
print(a)

