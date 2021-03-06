import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 本节在 try_activation_func.py 的基础上, 做可视化
# tensorboard 启动
# tensorboard --host 192.168.2.112 --logdir='logs/'

# y = Wx
# y = AF(Wx)
# relu, sigmoid, tanh
# 注意这些激励函数是要可以微分的
# 梯度爆炸, 梯度消失

# 少量层结构
# CNN relu
# RNN relu tanh

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    '''
    添加一个神经层
    '''
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weithts'):
            # 定义一个矩阵 in_size行,out_size列矩阵
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            # 
            # tf.histogram_summary(layer_name + "/Weights", Weights)
            tf.summary.histogram(layer_name + "/Weights", Weights)

        with tf.name_scope('biases'):
            # 列表
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            # tf.histogram_summary(layer_name + "/biases", biases)
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope('Wx_plus_bias'):
            Wx_plus_bias = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_bias
        else:
            outputs = activation_function(Wx_plus_bias)
            # tf.histogram_summary(layer_name + "/outputs", outputs)
            tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs


# 定义数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 输入层
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 隐藏层 hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# output
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope("loss"):
    # 损失函数
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

    # 原始版本 tf.scalar_summary("loss", loss)
    tf.summary.scalar("loss", loss)

with tf.name_scope('train'):
    # 学习效率, 一般小于1
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 定义了变量后, 一定要用下面的
# old -> tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()

# 合并 summary
# 原始版本 merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()

# 输出到日志
'''
不推荐使用SummaryWriter （来自tensorflow.python.training.summary_io），
将在2016-11-30之后删除。 
更新说明： 
请切换到tf.summary.FileWriter接口和行为是相同的; 这只是一个重命名。 
'''
# writer = tf.train.SummaryWriter("logs/",sess.graph)
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)

