from __future__ import print_function

import enum
import numpy as np
import sys
import tensorflow as tf
import time


class Model(object):
    class Phase(enum.Enum):
        FETCH = 0
        FUNC = 1

    def __init__(self, global_step, sess=None):
        self.global_step = tf.to_int32(global_step)
        if sess is None:
            sess = tf.get_default_session()
        self.sess = sess
        self.output_values = dict()

    def get_value(self, value, feed_dict=None):
        return self.sess.run(value, feed_dict=feed_dict)

    def get_callback(self, phase, callbacks, step, end_step):
        for callback in callbacks:
            run = callback.get('run', True)
            interval = callback.get('interval', 1)
            always = callback.get('always', False)
            fetch = callback.get('fetch', dict())
            func = callback.get('func', None)

            if run:
                on = (interval > 0 and (step + 1) % interval == 0) or (interval == -1 and step + 1 == end_step)
                if phase == Model.Phase.FETCH:
                    if on or always:
                        yield (fetch, func)
                elif phase == Model.Phase.FUNC:
                    if on and callable(func):
                        yield (fetch, func)

    def train(self, iteration, feed_dict=dict(), callbacks=list()):
        self.feed(
            current_step=self.get_value(self.global_step),
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=callbacks)

    def test(self, iteration, feed_dict=dict(), callbacks=list()):
        self.feed(
            current_step=0,
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=callbacks)

    def feed(self, current_step, iteration, feed_dict, callbacks):
        end_step = current_step + iteration

        for step in xrange(current_step, end_step):
            outputs = dict()
            for (fetch, func) in self.get_callback(Model.Phase.FETCH, callbacks, step, end_step):
                outputs.update(fetch)

            start_time = time.time()
            values = self.sess.run(outputs.values(), feed_dict=feed_dict)
            duration = time.time() - start_time

            self.output_values = dict(zip(outputs.keys(), values))
            for (fetch, func) in self.get_callback(Model.Phase.FUNC, callbacks, step, end_step):
                func(
                    fetch={key: self.output_values[key] for key in fetch},
                    duration=duration,
                    step=step,
                    end_step=end_step)

    def display(self, fetch, duration, step, end_step, precision=4, begin='', end='', **kwargs):
        step_length = np.floor(np.log10(end_step) + 1).astype(np.int32)
        format_str = '%%s %%%dd/%%%dd (%%.3f s)' % (step_length, step_length)
        print(format_str % (begin, step + 1, end_step, duration), end='')

        np.set_printoptions(precision=precision)
        for (key, value) in fetch.iteritems():
            value_str = ('%%.%df' % precision if value.ndim == 0 else '%s') % value
            print(', %s: %s' % (key, value_str), end='')
        print('', end=end)
        sys.stdout.flush()

    def display_train(self, **kwargs):
        self.display(begin='Train', end='\n', **kwargs)

    def display_test(self, step, end_step, **kwargs):
        self.display(step=step, end_step=end_step, begin='\033[2K\rTest', end='', **kwargs)
        if step + 1 == end_step:
            print('')

    def summary(self, fetch, summary_writer, **kwargs):
        for (key, value) in fetch.iteritems():
            summary_writer.add_summary(value, global_step=self.get_value(self.global_step))

    def save(self, saver, saver_kwargs=dict(), **kwargs):
        saver.save(self.sess, global_step=self.get_value(self.global_step), **saver_kwargs)
        print('[ Model saved ]')
