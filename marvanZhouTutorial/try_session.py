import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

# np.dot(m1,m2)
product = tf.matmul(matrix1,matrix2)
sess = tf.Session()


