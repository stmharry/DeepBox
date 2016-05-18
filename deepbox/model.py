from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


class Model(object):
    def __init__(self, global_step, sess=None):
        self.global_step = global_step
        self.sess = tf.get_default_session() if sess is None else sess

    def get_value(self, value, feed_dict=None):
        return self.sess.run(value, feed_dict=feed_dict)

    def get_callback(self, callbacks):
        for callback in callbacks:
            yield (
                callback.get('run', True),
                callback.get('interval', 1),
                callback.get('fetch', {}),
                callback.get('func', None))

    def is_run(self, step, interval, end_step):
        return (interval > 0 and (step + 1) % interval == 0) or (interval == -1 and (step + 1) == end_step)

    def train(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(int(self.get_value(self.global_step)), iteration, feed_dict, callbacks)

    def test(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(0, iteration, feed_dict, callbacks)

    def feed(self, current_step, iteration, feed_dict, callbacks):
        end_step = current_step + iteration
        for i in xrange(current_step, end_step):
            output_dict = {}
            for (run, interval, fetch, func) in self.get_callback(callbacks):
                if run and self.is_run(i, interval, end_step):
                    output_dict.update(fetch)

            values = self.sess.run(output_dict.values(), feed_dict=feed_dict)
            output_value_dict = dict(zip(output_dict.keys(), values))

            for (run, interval, fetch, func) in self.get_callback(callbacks):
                if run and self.is_run(i, interval, end_step) and callable(func):
                    func(dict(
                        current_step=i,
                        end_step=end_step,
                        fetch=fetch,
                        output_value_dict=output_value_dict))

    def display(self, callback_dict, begin='', end='', onEnd=False):
        (current_step, end_step, fetch, output_value_dict) = (
            callback_dict['current_step'],
            callback_dict['end_step'],
            callback_dict['fetch'],
            callback_dict['output_value_dict'])

        step_length = np.floor(np.log10(end_step) + 1).astype(np.int32)
        format_str = '%%s %%%dd/%%%dd' % (step_length, step_length)
        print(format_str % (begin, current_step + 1, end_step), end='')

        if not (onEnd and current_step + 1 < end_step):
            np.set_printoptions(precision=8)
            for key in fetch.keys():
                output_value_str = ('%.8f' if output_value_dict[key].ndim == 0 else '%s') % output_value_dict[key]
                print(', %s: %s' % (key, output_value_str), end='')
        print('', end=end)
        sys.stdout.flush()

    def summary(self, callback_dict, summary_writer):
        (fetch, output_value_dict) = (
            callback_dict['fetch'],
            callback_dict['output_value_dict'])

        for key in fetch.keys():
            summary_writer.add_summary(output_value_dict[key], global_step=self.get_value(self.global_step))

    def save(self, callback_dict, saver, **kwargs):
        saver.save(self.sess, global_step=int(self.get_value(self.global_step)), **kwargs)
        print('[ Model saved ]')
