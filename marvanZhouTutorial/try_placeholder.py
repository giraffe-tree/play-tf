import tensorflow as tf

# 本节主要讲 placeholder

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 原教程中为 mul, 我使用的版本为 multiply
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
