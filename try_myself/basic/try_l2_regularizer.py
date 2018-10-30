import tensorflow as tf
import numpy as np


# 获取一层神经网络边上的权重
def get_weight(shape, ratio):
    v1 = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    #
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(ratio)(v1))
    return v1


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8

# 定义每一层网络中的节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 最深层节点
cur_layer = x
# 当前层的节点数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # 下一层节点数
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 更新为下一层节点数
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection("losses", mse_loss)

loss = tf.add_n(tf.get_collection("losses"))

# 学习率
global_step = tf.Variable(200)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 随机数生成一个模拟数据集
rdm = np.random.RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)

Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]
# print(X)
# print(Y)

# 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataSet_size
        end = min(start + batch_size, dataSet_size)
        result = sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start: end]})
        if i % 100 == 0:
            print(result)
