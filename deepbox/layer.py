import tensorflow as tf
import deepbox.image_util as image_util


def avg_pool(value, name, size, stride=None, padding='SAME'):
    with tf.variable_scope(name):
        if stride is None:
            stride = size
        value = tf.nn.avg_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding, name='pool')
    return value


def max_pool(value, name, size, stride=None, padding='SAME'):
    with tf.variable_scope(name):
        if stride is None:
            stride = size
        value = tf.nn.max_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding, name='pool')
    return value


def bn(value, name, mean=None, variance=None, scale=None, offset=None, trainable=True, collections=None):
    in_channel = image_util.get_channel(value)

    with tf.variable_scope(name):
        if mean is None:
            mean_initializer = tf.zeros_initializer
        else:
            mean_initializer = tf.constant_initializer(mean)

        mean_variable = tf.get_variable(
            'mean',
            shape=(in_channel,),
            initializer=mean_initializer,
            trainable=trainable,
            collections=collections)

        if variance is None:
            variance_initializer = tf.ones_initializer
        else:
            variance_initializer = tf.constant_initializer(variance)

        variance_variable = tf.get_variable(
            'variance',
            shape=(in_channel,),
            initializer=variance_initializer,
            trainable=trainable,
            collections=collections)

        if scale is None:
            scale_initializer = tf.ones_initializer
        else:
            scale_initializer = tf.constant_initializer(scale)

        scale_variable = tf.get_variable(
            'scale',
            shape=(in_channel,),
            initializer=scale_initializer,
            trainable=trainable,
            collections=collections)

        if offset is None:
            offset_initializer = tf.zeros_initializer
        else:
            offset_initializer = tf.constant_initializer(offset)

        offset_variable = tf.get_variable(
            'offset',
            shape=(in_channel,),
            initializer=offset_initializer,
            trainable=trainable,
            collections=collections)
