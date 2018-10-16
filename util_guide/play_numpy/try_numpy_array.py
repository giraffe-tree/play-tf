import numpy as np

# https://github.com/cs231n/cs231n.github.io/blob/b974c7cbd184f4fdbf1b4d7c3c60b76e16d67300/python-numpy-tutorial.md#numpy

a = np.array([1,2,3])
# ndarray
a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"

print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(type(b))            # Prints "<class 'numpy.ndarray'>"

print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"




