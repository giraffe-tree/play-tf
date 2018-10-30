import tensorflow as tf

matrix1 = tf.constant([[3, 3]], name="hello")
matrix2 = tf.constant([[2], [2]], name="world")

# np.dot(m1,m2)
product = tf.matmul(matrix1, matrix2)

config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)

sess = tf.Session(config=config)
result = product.eval(session=sess)

print(result, type(result))
