import tensorflow as tf

CHANNEL = 3


def get_shape(value):
    return tuple(value.get_shape().as_list())


def get_size(value):
    return get_shape(value)[1:3]


def get_channel(value):
    return get_shape(value)[3]


def random_int(lower, upper):
    return tf.to_int32(lower + tf.random_uniform(()) * tf.to_float(upper - lower))


def convert_to_rgb(value):
    shape = tf.shape(value)
    channel = shape[2]

    value = tf.cond(
        tf.equal(channel, 1),
        lambda: tf.image.grayscale_to_rgb(value),
        lambda: value)

    return value


def random_resize(value, size_range):
    new_shorter_size = random_int(size_range[0], size_range[1] + 1)

    shape = tf.shape(value)
    height = shape[0]
    width = shape[1]
    height_smaller_than_width = tf.less(height, width)

    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_size, new_shorter_size * width / height),
        lambda: (new_shorter_size * height / width, new_shorter_size))

    value = tf.expand_dims(value, 0)
    value = tf.image.resize_bilinear(value, tf.pack(new_height_and_width))
    value = tf.squeeze(value, [0])
    return value


def random_crop(value, size):
    shape = tf.shape(value)
    height = shape[0]
    width = shape[1]

    offset_height = random_int(0, height - size + 1)
    offset_width = random_int(0, width - size + 1)

    value = tf.slice(
        value,
        tf.pack([offset_height, offset_width, 0]),
        tf.pack([size, size, -1]))
    value.set_shape([size, size, CHANNEL])
    return value


def random_flip(value):
    value = tf.image.random_flip_left_right(value)
    value = tf.image.random_flip_up_down(value)
    return value


def random_adjust(value, max_delta, contrast_lower, contrast_upper):
    value = tf.image.random_brightness(value, max_delta=max_delta)
    value = tf.image.random_contrast(value, lower=contrast_lower, upper=contrast_upper)
    return value
