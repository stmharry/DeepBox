import tensorflow as tf

CHANNEL = 3


def get_shape(value):
    return tuple(value.get_shape().as_list())


def get_size(value):
    return get_shape(value)[1:3]


def get_channel(value):
    return get_shape(value)[3]


def random(lower, upper):
    return lower + tf.random_uniform(()) * tf.to_float(upper - lower)


def random_int(lower, upper):
    return tf.to_int32(random(lower, upper))


def random_resize(value, size_range, keep_aspect_ratio=True):
    new_shorter_size = random_int(size_range[0], size_range[1] + 1)

    shape = tf.shape(value)
    height = shape[0]
    width = shape[1]
    height_smaller_than_width = tf.less(height, width)

    if ratio_range is None:


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


def random_flip(value, horizontal=True, vertical=False):
    if horizontal:
        value = tf.image.random_flip_left_right(value)
    if vertical:
        value = tf.image.random_flip_up_down(value)
    return value


def random_adjust_rgb(value, max_delta=63, contrast_range=(0.5, 1.5)):
    value = tf.image.random_brightness(value, max_delta=max_delta)
    value = tf.image.random_contrast(value, lower=contrast_range[0], upper=contrast_range[1])
    return value


def random_adjust_hsv(value, h_offset_range=(-0.1, 0.1), sv_power_range=(0.25, 4), sv_scale_range=(0.7, 1.4), sv_offset_range=(-0.1, 0.1)):
    (h, s, v) = tf.split(split_dim=2, num_split=3, value=value)

    def power_scale_offset(v, power_range, scale_range, offset_range):
        v = tf.pow(v, random(power_range[0], power_range[1]))
        v = v * random(scale_range[0], scale_range[1])
        v = v + random(offset_range[0], offset_range[1])
        return v

    h = h + random(h_offset_range[0], h_offset_range[1])
    s = power_scale_offset(s, sv_power_range, sv_scale_range, sv_offset_range)
    v = power_scale_offset(v, sv_power_range, sv_scale_range, sv_offset_range)

    value = tf.concat(concat_dim=2, values=[h, s, v])
    return value
