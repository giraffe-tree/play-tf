import tensorflow as tf

# tensor 的默认类型
# 整型 int32, 小数 float32

matrix1 = tf.constant([[3, 3]], name="hello")
matrix2 = tf.constant([[2], [2]], name="world")
print(matrix1, type(matrix1))

# np.dot(m1,m2)
product = tf.matmul(matrix1, matrix2)
print(product, type(product))

# method 1
# result = tf.Session().run(product)

# method 2
sess = tf.Session()
result = product.eval(session=sess)

print(result, type(result))
