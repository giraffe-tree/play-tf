import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 +0.3

# create tensorflow structure start 

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

# 预测
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

# 优化器 0.5 为学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# old -> tf.initialize_all_variables()
init = tf.global_variables_initializer()

# create tensorflow structure end

sess = tf.Session()
sess.run(init)

for step in range(200):
	sess.run(train)
	if step%20==0:
		print(step,sess.run(Weights),sess.run(biases))


