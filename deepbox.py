from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

_EPSILON = 1e-7
_EPOCH = 1
_BATCH_SIZE = 256
_REPORT_PER = 1
_BAR_LENGTH = 20

TRAIN = 0
VAL = 1
TEST = 2


class Dataset(object):

    @staticmethod
    def split(X, ind):
        return [np.split(x, ind) for x in X]

    @staticmethod
    def normalize(X):
        _X = np.concatenate(X)
        m = np.nanmean(_X, 0)
        s = np.nanstd(_X, 0) + _EPSILON

        return [(x - m) / s for x in X]

    @staticmethod
    def one_hot(Y):
        _Y = np.concatenate(Y)
        y0 = np.unique(_Y[~np.isnan(_Y)])

        _Y = []
        for y in Y:
            _y = (y[:, None] == y0).astype(float)
            _y[np.isnan(y), :] = np.NaN
            _Y += [_y]

        return (_Y, y0)

    @staticmethod
    def get_sample_num(feed_dict):
        return max([v.shape[0] for v in feed_dict.values() if isinstance(v, np.ndarray)]) if feed_dict else 0

    @staticmethod
    def get_batch(feed_dict, sample_num, batch_size, total_batch_num):
        for batch_num in range(total_batch_num):
            batch = {}
            min_batch_index = batch_num * batch_size
            max_batch_index = min((batch_num + 1) * batch_size, sample_num)

            for (key, value) in feed_dict.iteritems():
                if isinstance(value, np.ndarray):
                    batch[key] = value[min_batch_index:max_batch_index]
                else:
                    batch[key] = value
            yield (batch, batch_num, min_batch_index, max_batch_index)

    @staticmethod
    def permute(feed_dict):
        state = np.random.get_state()
        for value in feed_dict.values():
            if isinstance(value, np.ndarray):
                np.random.set_state(state)
                np.random.shuffle(value)

class Model(object):

    @staticmethod
    def print_report(phase, epoch_num, total_epoch_num, progress, keys, values):
        phase_str = ['train', 'val', 'test']
        epoch_length = np.floor(np.log10(total_epoch_num)).astype(int) + 1
        progress_length = int(progress * _BAR_LENGTH)
        neg_progress_length = _BAR_LENGTH - progress_length

        if phase != VAL:
            print('\rEpoch: %*d/%*d [%s%s]  ' % (
                epoch_length, epoch_num, epoch_length, total_epoch_num, '=' * progress_length, '-' * neg_progress_length), end='')
        if (phase != VAL) or ((phase == VAL) and (progress == 1)):
            if (len(keys) != 0) and (len(values) != 0):
                for (k, v) in zip(keys, values):
                    print('%s: %.4f  ' %
                          (phase_str[phase] + '-' + k, v), end='')
        print('\033[K', end='')
        sys.stdout.flush()

    @staticmethod
    def get_shape(x):
        return tuple([int(s) if s is not None else None for s in x.get_shape().as_list()])

    @staticmethod
    def get_batch_shape(x):
        return tuple([int(s) if s is not None else None for s in x.get_shape().as_list()[1:]])

    @staticmethod
    def get_size(x):
        return np.prod(Model.get_batch_shape(x))

    @staticmethod
    def full_shape(shape):
        assert len(shape) == 2
        return (1,) + shape + (1,)

    @staticmethod
    def weight(shape, name='', **kwargs):
        return tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(6 / np.sum(shape))), name='weight-' + name, **kwargs)

    @staticmethod
    def bias(shape, name='', **kwargs):
        return tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(6 / np.sum(shape))), name='bias-' + name, **kwargs)

    @staticmethod
    def dense(x, output_dim, **kwargs):
        input_shape = Model.get_batch_shape(x)
        assert len(input_shape) == 1
        return tf.matmul(x, Model.weight((input_shape[0], output_dim), **kwargs)) + Model.bias((output_dim,), **kwargs)

    @staticmethod
    def dense_relu_drop(x, d, output_dim, **kwargs):
        if d == 1.0:
            return tf.nn.relu6(Model.dense(x, output_dim, **kwargs))
        else:
            return tf.nn.dropout(tf.nn.relu6(Model.dense(x, output_dim, **kwargs)), d)

    @staticmethod
    def dense_sequential(x, d, arch):
        h = x
        for output_dim in arch:
            h = Model.dense_relu_drop(h, d, output_dim)
        return h

    @staticmethod
    def flatten(x):
        return tf.reshape(x, (-1, Model.get_size(x)))

    @staticmethod
    def max_pool(x, shape, strides=(), padding='VALID'):
        input_shape = Model.get_batch_shape(x)
        assert len(input_shape) == 3
        if not strides:
            strides = shape
        return tf.nn.max_pool(x, Model.full_shape(shape), Model.full_shape(strides), padding)

    @staticmethod
    def conv2d(x, num_kernel, shape, strides=(), padding='VALID'):
        input_shape = Model.get_batch_shape(x)
        assert len(input_shape) == 3
        if not strides:
            strides = shape
        return tf.nn.conv2d(x, Model.weight(shape + (input_shape[2], num_kernel)), Model.full_shape(strides), padding) + Model.bias((num_kernel,))

    @staticmethod
    def embedding(x, shape):
        input_shape = Model.get_batch_shape(x)
        assert len(input_shape) == 1
        with tf.device("/cpu:0"):
            return tf.reshape(tf.nn.embedding_lookup(Model.weight(shape), x), (-1, shape[1]))

    #
    def __init__(self):
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        self.val_flag = False
        self.args = [{
            'inputs': {},
            'outputs': {},
            'updates': {},
            'kwargs': {'epoch': _EPOCH, 'batch_size': _BATCH_SIZE, 'report_per': _REPORT_PER}} for _ in xrange(3)]
        self.output_values = [[] for _ in xrange(3)]

    def init(self, excludes=[]):
        self.sess.run(tf.initialize_variables(
            [v for v in tf.all_variables() if v not in excludes]))

    def set_train(self, *args, **kwargs):
        self.set(TRAIN, *args, **kwargs)

    def set_val(self, *args, **kwargs):
        self.set(VAL, *args, **kwargs)

    def set_test(self, *args, **kwargs):
        self.set(TEST, *args, **kwargs)

    def set(self, phase, inputs=None, outputs=None, updates=None, **kwargs):
        if phase == VAL:
            self.val_flag = True
        for (var, attr) in zip([inputs, outputs, updates], ['inputs', 'outputs', 'updates']):
            if var is not None:
                self.args[phase][attr] = var
        self.args[phase]['kwargs'].update(kwargs)

    #

    def train(self, **kwargs):
        self.set(TRAIN, **kwargs)
        return self.feed(TRAIN)

    def test(self, **kwargs):
        self.set(TEST, **kwargs)
        return self.feed(TEST)

    #

    def feed(self, phase):
        args = self.args[phase]
        (inputs, outputs, updates, kwargs) = (
            args['inputs'], args['outputs'], args['updates'], args['kwargs'])
        (total_epoch_num, batch_size, report_per) = (
            kwargs['epoch'], kwargs['batch_size'], kwargs['report_per'])

        sample_num = Dataset.get_sample_num(inputs)
        batch_size = sample_num if batch_size == -1 else batch_size
        total_batch_num = int(np.ceil(float(sample_num) / batch_size))
        self.output_values[phase] = []

        key_shape = [Model.get_shape(key) for key in outputs.keys()]
        key_form = [bool(shape) and ((shape[0] is None) or (
            shape[0] == batch_size)) for shape in key_shape]
        key_show = [not form and value is not None for (
            form, value) in zip(key_form, outputs.values())]

        for epoch_num in xrange(total_epoch_num):
            if phase == TRAIN:
                Dataset.permute(inputs)

            value_epoch = dict([
                (key, np.zeros((sample_num,) + shape[1:]) if form else np.zeros((total_batch_num,) + shape))
                for (key, shape, form) in zip(outputs.keys(), key_shape, key_form)])

            for (batch, batch_num, min_batch_index, max_batch_index) in Dataset.get_batch(inputs, sample_num, batch_size, total_batch_num):
                value_batch = self.sess.run(outputs.keys() + updates.keys(), feed_dict=batch)[:len(outputs)]

                for (key, value, form) in zip(outputs.keys(), value_batch, key_form):
                    if form:
                        value_epoch[key][min_batch_index:max_batch_index] = value
                    else:
                        value_epoch[key][batch_num] = value

                if np.mod(epoch_num + 1, report_per) == 0:
                    Model.print_report(
                        phase,
                        epoch_num + 1,
                        total_epoch_num,
                        (batch_num + 1.) / total_batch_num,
                        [value for (show, value) in zip(key_show, outputs.values()) if show],
                        [np.sum(value_epoch[key]) / (batch_num + 1) for (show, key) in zip(key_show, outputs.keys()) if show])

            self.output_values[phase] += [value_epoch]
            if np.mod(epoch_num + 1, report_per) == 0:
                if (phase == TRAIN) and self.val_flag:
                    self.feed(VAL)
                if phase != VAL:
                    print('')

    def save(self, path, step=None):
        if step is None:
            self.saver.save(self.sess, path)
        else:
            self.saver.save(self.sess, path, global_step=step)

    def load(self, path):
        state = tf.train.get_checkpoint_state(path)
        if state and state.model_checkpoint_path:
            self.saver.restore(self.sess, state.model_checkpoint_path)
