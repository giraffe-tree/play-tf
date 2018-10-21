import tensorflow as tf
import numpy as np

# 本节主要讲解变量/常量

state = tf.Variable(0,name="counter")
# print(state.name)
one = tf.constant(1)
state2 = tf.add(state,one)
update = tf.assign(state,state2)

# 定义了变量后, 一定要用下面的
# old -> tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))


