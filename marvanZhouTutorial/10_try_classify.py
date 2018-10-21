import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 本节讲述 classification 分类
'''
来自: https://www.youtube.com/watch?v=aNjdw9w_Qyc
机器学习中的监督学习(supervised learning)问题大部分可以分成 Regression (回归)
和 Classification(分类) 这两种. Tensorflow 也可以做到这个. 回归是说我要预测的
值是一个连续的值,比如房价,汽车的速度,飞机的高度等等.而分类是指我要把东西分成几类
,比如猫狗猪牛等等. 我们之前的教程都是在用 regression 来教学的,这一次就介绍了如
何用 Tensorflow 做 classification.
'''

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


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


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,n_layer=1, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))


