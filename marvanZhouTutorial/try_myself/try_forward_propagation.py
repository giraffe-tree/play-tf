import tensorflow as tf

# 正太分布
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=0))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 特征向量输入
x = tf.constant([[0.7,0.9]])

# 均匀分布
# y = tf.random_uniform([2,3])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

init = tf.global_variables_initializer()

sess = tf.Session()
result = sess.run(init)

print(sess.run(a))
print(sess.run(y))