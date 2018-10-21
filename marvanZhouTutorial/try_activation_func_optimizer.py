import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 线性优化
#  SGD 随机梯度下降
# stochastic gradient descent 随机梯度下降

# 加速神经网络
# momentum 动量
# adagrad 更新学习率
# RMSProp -> adam 

# tensenflow optimizer 优化器
'''
tf.train.AdamOptimizer
tf.train.MomentumOptimizer
'''


def add_layer(inputs, in_size, out_size, activation_function=None):
    '''
    添加一个神经层
    '''
    # 定义一个矩阵 in_size行,out_size列矩阵
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 列表
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_bias = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_bias
    else:
        outputs = activation_function(Wx_plus_bias)
    return outputs


# 定义数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# 隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function=None)

# 损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

# 学习效率, 一般小于1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 定义了变量后, 一定要用下面的
# old -> tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train_step,feed_dict={xs: x_data,ys:y_data})
    if i % 50 ==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))



