from __future__ import print_function

import tensorflow as tf

EPSILON = 1e-9


class NetworkNormalization(object):
    def __init__(self, axes=(0, 1, 2, 3), tolerance=0.1, max_iteration=10):
        self.axes = axes
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.collection = list()

    def register(self, source, targets):
        (_, variance) = tf.nn.moments(source, axes=self.axes)
        condition = tf.greater(tf.abs(variance - 1), self.tolerance)

        target_assigns = [
            target.assign(tf.cond(
                condition,
                lambda: target / tf.sqrt(variance),
                lambda: target)) for target in targets]

        self.collection.append(dict(
            source=source,
            targets=targets,
            variance=variance,
            condition=condition,
            target_assigns=target_assigns))

    def run(self, sess=None, feed_dict=None):
        if sess is None:
            sess = tf.get_default_session()

        print('[ Network Normalization ]')
        for item in self.collection:
            print('%s -> ' % item['source'].name, end='')
            for target in item['targets']:
                print('%s ' % target.name, end='')
            print('')
            for iteration in xrange(self.max_iteration):
                (variance, condition) = sess.run([item['variance'], item['condition']] + item['target_assigns'], feed_dict=feed_dict)[:2]
                print('Iteration %d, variance: %.4f' % (iteration, variance))
                if not condition:
                    break


def increment_variable(init=0):
    num = tf.Variable(init, dtype=tf.float32, trainable=False)
    num_ = num + 1
    with tf.control_dependencies([num.assign(num_)]):
        return tf.identity(num_)


def moving_average(value, window):
    value = tf.to_float(value)
    shape = value.get_shape()

    queue_init = tf.zeros(tf.TensorShape(window).concatenate(shape))
    total_init = tf.zeros(shape)
    num_init = tf.constant(0, dtype=tf.float32)

    queue = tf.FIFOQueue(window, [tf.float32], shapes=[shape])
    total = tf.Variable(total_init, trainable=False)
    num = tf.Variable(num_init, trainable=False)

    init = tf.cond(
        tf.equal(queue.size(), 0),
        lambda: tf.group(
            queue.enqueue_many(queue_init),
            total.assign(total_init),
            num.assign(num_init)),
        lambda: tf.no_op())

    with tf.control_dependencies([init]):
        total_ = total + value - queue.dequeue()
        num_ = num + 1
        value_averaged = total_ / (tf.minimum(num_, window) + EPSILON)

        with tf.control_dependencies([queue.enqueue([value]), total.assign(total_), num.assign(num_)]):
            return tf.identity(value_averaged)


def exponential_moving_average(value, decay=0.99, num_updates=None, init=None):
    value = tf.to_float(value)
    shape = value.get_shape()

    if init is None:
        init = tf.zeros(shape)
        if num_updates is not None:
            decay = tf.minimum(decay, (1 + num_updates) / (10 + num_updates))

    value_averaged = tf.Variable(init, trainable=False)
    value_averaged_ = decay * value_averaged + (1 - decay) * value
    with tf.control_dependencies([value_averaged.assign(value_averaged_)]):
        return tf.identity(value_averaged_)
