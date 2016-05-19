import tensorflow as tf

EPSILON = 1e-9


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
