import tensorflow as tf

'''
pipenv --pypi-mirror https://mirrors.aliyun.com/pypi/simple/ install -v tensorflow-gpu
reference: 
    https://pipenv.readthedocs.io/en/latest/#cmdoption-pipenv-install-extra-index-url
'''

# np.dot(m1,m2)
# with tf.device('/device:GPU:0'):
#     product = tf.matmul(matrix1, matrix2)

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
