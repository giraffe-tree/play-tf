import tensorflow as tf
import numpy as np

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")

# 一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 单层前向传播, w1 为要计算的参数
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))

# 设置衰减的学习率
# learning_rate, global_step, decay_steps, decay_rate, staircase = False
# learning_rate * decay_rate ^ (global_step / decay_steps)
# lr = 0.1
# for i in range(100, 300):
#     print(lr)
#     lr = lr * 0.96 ** (i / 100)

global_step = tf.Variable(200)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 随机数生成一个模拟数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]
# print(X)
# print(Y)

# 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start: end]})
        if i % 100 == 0:
            print(sess.run(learning_rate))
            print(sess.run(global_step))
            print(sess.run(w1))
