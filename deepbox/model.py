from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


class Model(object):
    def __init__(self, global_step, sess=None):
        self.global_step = tf.to_int32(global_step)
        if sess is None:
            sess = tf.get_default_session()
        self.sess = sess

    def get_value(self, value, feed_dict=None):
        return self.sess.run(value, feed_dict=feed_dict)

    def get_callback(self, callbacks, step, end_step):
        for callback in callbacks:
            run = callback.get('run', True)
            interval = callback.get('interval', 1)
            fetch = callback.get('fetch', {})
            func = callback.get('func', None)

            if run and ((interval > 0 and (step + 1) % interval == 0) or (interval == -1 and step + 1 == end_step)) and callable(func):
                yield (fetch, func)

    def train(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(
            current_step=self.get_value(self.global_step),
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=callbacks)

    def test(self, iteration, feed_dict={}, callbacks=[]):
        self.feed(
            current_step=0,
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=callbacks)

    def feed(self, current_step, iteration, feed_dict, callbacks):
        end_step = current_step + iteration

        for step in xrange(current_step, end_step):
            output_dict = {}
            for (fetch, func) in self.get_callback(callbacks, step, end_step):
                output_dict.update(fetch)

            values = self.sess.run(output_dict.values(), feed_dict=feed_dict)
            output_value_dict = dict(zip(output_dict.keys(), values))

            for (fetch, func) in self.get_callback(callbacks, step, end_step):
                fetch_val = {key: output_value_dict[key] for key in fetch}
                func(fetch=fetch_val, step=step, end_step=end_step)

    def display(self, fetch, step, end_step, begin='', end='', on_end=False):
        step_length = np.floor(np.log10(end_step) + 1).astype(np.int32)
        format_str = '%%s %%%dd/%%%dd' % (step_length, step_length)
        print(format_str % (begin, step + 1, end_step), end='')

        if (not on_end) or (on_end and step + 1 == end_step):
            np.set_printoptions(precision=8)
            for (key, value) in fetch.iteritems():
                value_str = ('%.8f' if value.ndim == 0 else '%s') % value
                print(', %s: %s' % (key, value_str), end='')
        print('', end=end)
        sys.stdout.flush()

    def summary(self, fetch, step, end_step, summary_writer):
        for (key, value) in fetch.iteritems():
            summary_writer.add_summary(value, global_step=self.get_value(self.global_step))

    def save(self, fetch, step, end_step, saver, saver_kwargs):
        saver.save(self.sess, global_step=self.get_value(self.global_step), **saver_kwargs)
        print('[ Model saved ]')
