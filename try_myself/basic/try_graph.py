import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)

    update = tf.assign(x1, x1 + x2)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(update)
        print(sess.run(x1))

g2 = tf.Graph()
with g2.as_default():
    x1 = tf.Variable(10.0)
    x2 = tf.Variable(20.0)

    update = tf.assign(x1, x1 + x2)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(update)
        print(sess.run(x1))

