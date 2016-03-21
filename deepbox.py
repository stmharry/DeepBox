from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

_EPSILON = 1e-7
_EPOCH = 1
_BATCH_SIZE = 256
_REPORT_PER = 1
_BAR_LENGTH = 20

_PHASES = ['train', 'val', 'test']


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
    def guess_sample_num(feed_dict):
        if feed_dict:
            sample_nums = [v.shape[0]
                           for v in feed_dict.values()
                           if isinstance(v, np.ndarray)]
            return max(sample_nums)
        else:
            return 0

    @staticmethod
    def get_batch(feed_dict, sample_num, batch_size, total_batch_num):
        for batch_num in range(total_batch_num):
            batch = {}
            min_batch_index = batch_num * batch_size
            max_batch_index = min((batch_num + 1) * batch_size, sample_num)
            this_batch_size = max_batch_index - min_batch_index

            for (key, value) in feed_dict.iteritems():
                if isinstance(value, np.ndarray):
                    batch[key] = value[min_batch_index:max_batch_index]
                else:
                    batch[key] = value
            yield (batch, batch_num, this_batch_size, min_batch_index, max_batch_index)

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
        epoch_length = np.floor(np.log10(total_epoch_num)).astype(int) + 1
        progress_length = int(progress * _BAR_LENGTH)
        neg_progress_length = _BAR_LENGTH - progress_length

        if phase != 'val':
            print('\rEpoch: %*d/%*d [%s%s]  ' % (
                epoch_length, epoch_num, epoch_length, total_epoch_num, '=' * progress_length, '-' * neg_progress_length), end='')
        if (phase != 'val') or ((phase == 'val') and (progress == 1)):
            if (len(keys) != 0) and (len(values) != 0):
                for (k, v) in zip(keys, values):
                    print('%s: %.4f  ' %
                          (phase + '-' + k, v), end='')
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
        return tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(6. / np.sum(shape))), name='weight-' + name, **kwargs)

    @staticmethod
    def bias(shape, name='', **kwargs):
        return tf.Variable(tf.fill(shape, value=0.1), name='bias-' + name, **kwargs)

    @staticmethod
    def dense(x, output_dim, **kwargs):
        input_shape = Model.get_batch_shape(x)
        assert len(input_shape) == 1
        return tf.matmul(x, Model.weight((input_shape[0], output_dim), **kwargs)) + Model.bias((output_dim,), **kwargs)

    @staticmethod
    def dense_relu(x, output_dim, **kwargs):
        return tf.nn.relu6(Model.dense(x, output_dim, **kwargs))

    @staticmethod
    def dense_relu_drop(x, d, output_dim, **kwargs):
        if d == 1.0:
            return Model.dense_relu(x, output_dim, **kwargs)
        else:
            return tf.nn.dropout(Model.dense_relu(x, output_dim, **kwargs), d)

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
        if tf.all_variables():
            self.saver = tf.train.Saver()
        self.sess = tf.Session()

        self.val_flag = False
        self.args = dict(zip(_PHASES, [{
            'inputs': {},
            'outputs': {},
            'updates': {},
            'sample_num': -1,
            'batch_size': _BATCH_SIZE,
            'epoch': _EPOCH,
            'report_per': _REPORT_PER,
            'after_batch': [],
            'after_epoch': []} for _ in xrange(len(_PHASES))]))
        self.output_values = dict(zip(_PHASES, [[] for _ in xrange(len(_PHASES))]))

    def init(self, excludes=[]):
        self.sess.run(tf.initialize_variables(
            [v for v in tf.all_variables() if v not in excludes]))

    def set(self, phase, **kwargs):
        if phase == 'val':
            self.val_flag = True
        self.args[phase].update(kwargs)

    #

    def train(self, dry_run=False, **kwargs):
        self.set('train', **kwargs)
        if not dry_run:
            self.feed('train')

    def val(self, dry_run=False, **kwargs):
        assert dry_run
        self.set('val', **kwargs)

    def test(self, dry_run=False, **kwargs):
        self.set('test', **kwargs)
        if not dry_run:
            self.feed('test')

    #

    def feed(self, phase):
        args = self.args[phase]
        (inputs, outputs, updates, sample_num, batch_size, total_epoch_num, report_per, after_batch, after_epoch) = (
            args['inputs'],
            args['outputs'],
            args['updates'],
            args['sample_num'],
            args['batch_size'],
            args['epoch'],
            args['report_per'],
            args['after_batch'],
            args['after_epoch'])

        if sample_num == -1:
            sample_num = Dataset.guess_sample_num(inputs)
        if batch_size == -1:
            batch_size = sample_num
        total_batch_num = int(np.ceil(float(sample_num) / batch_size))
        self.output_values[phase] = []

        shape = {key: Model.get_shape(key) for key in outputs.keys()}
        for_batch = {key: bool(shape[key]) and ((shape[key][0] is None) or (shape[key][0] == batch_size)) for key in outputs.keys()}
        show = {key: (not for_batch[key]) and (value is not None) for (key, value) in outputs.iteritems()}
        # batch_shape = {key: ((sample_num,) + shape[key][1:]) if for_batch[key] else ((total_batch_num,) + shape[key]) for key in outputs.keys()}

        for epoch_num in xrange(total_epoch_num):
            if phase == 'train':
                Dataset.permute(inputs)

            # output_value_epoch = {key: np.zeros(shape[key]) for key in outputs.keys()}
            output_value_batch_list = {key: [] for key in outputs.keys()}

            for (batch, batch_num, this_batch_size, min_batch_index, max_batch_index) in Dataset.get_batch(inputs, sample_num, batch_size, total_batch_num):
                output_value_batch = self.sess.run(outputs.keys() + updates.keys(), feed_dict=batch)[:len(outputs)]

                for (key, output_value) in zip(outputs.keys(), output_value_batch):
                    if for_batch[key]:
                        output_value = output_value[:this_batch_size]
                        # output_value_epoch[key][min_batch_index:max_batch_index] = output_value_batch
                        output_value_batch_list[key] += [output_value]
                    else:
                        # output_value_epoch[key][batch_num] = output_value_batch
                        output_value_batch_list[key] += [output_value]

                if np.mod(epoch_num + 1, report_per) == 0:
                    Model.print_report(
                        phase,
                        epoch_num + 1,
                        total_epoch_num,
                        (batch_num + 1.) / total_batch_num,
                        [value for (key, value) in outputs.iteritems() if show[key]],
                        [output_value_batch_list[key][-1] for key in outputs.keys() if show[key]])
                        # [np.sum(output_value_epoch[key]) / (batch_num + 1) for (key, value) in outputs.iteritems() if show[key]])

                for callback in after_batch:
                    assert callable(callback)
                    callback(batch_num, dict(zip(outputs.keys, output_value_batch)))

            output_value_epoch = {}
            for key in outputs.keys():
                if for_batch[key]:
                    output_value_epoch[key] = np.concatenate(output_value_batch_list[key], axis=0)
                else:
                    output_value_epoch[key] = np.stack(output_value_batch_list[key], axis=0)
            self.output_values[phase] += [output_value_epoch]

            if np.mod(epoch_num + 1, report_per) == 0:
                if (phase == 'train') and self.val_flag:
                    self.feed('val')
                if phase != 'val':
                    print('')

            for callback in after_epoch:
                assert callable(callback)
                callback(epoch_num, output_value_epoch)

    def save(self, path, step=None):
        if step is None:
            self.saver.save(self.sess, path)
        else:
            self.saver.save(self.sess, path, global_step=step)

    def load(self, path):
        state = tf.train.get_checkpoint_state(path)
        if state and state.model_checkpoint_path:
            self.saver.restore(self.sess, state.model_checkpoint_path)
