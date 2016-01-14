import tensorflow as tf

from numpy import mean, std, unique, prod
from numpy import arange
from numpy import concatenate

_EPSILON = 1e-7

class Dataset(object):
    def __init__(self, **kwargs):
        (self.X, self.Y) = kwargs['data']
        (self.len_train, self.len_val, self.len_test) = kwargs['length']

        if 'normalize' in kwargs:
            if kwargs['normalize'] == True:
                self.X = Dataset.normalize(self.X)

        if 'one_hot' in kwargs:
            if kwargs['one_hot'] == True:
                (self.Y, self.y0) = Dataset.one_hot(self.Y)
            else:
                self.y0 = arange(self.Y.shape[1])
        
        self.input_dim = self.X.shape[1:]
        self.output_dim = self.Y.shape[1:]
    
    @staticmethod
    def normalize(X): 
        return (X - mean(X, 0)) / (std(X, 0) + _EPSILON)

    @staticmethod
    def one_hot(Y):
        y0 = unique(Y)
        return ((Y[:, None] == y0).astype(float), y0)

class Model(object):
    def __init__(self):
        self.inputs = []
        self.outputs = []

    @staticmethod
    def get_shape(x):
        return x.get_shape().as_list()[1:]

    @staticmethod
    def get_size(x):
        return prod(Model.get_shape(x))

    @staticmethod
    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    @staticmethod
    def dense(x, shape):
        return tf.matmul(x, Model.weight(shape)) + Model.bias(shape[-1:])
    
    @staticmethod
    def max_pool(x, shape, strides, padding):
        return tf.nn.max_pool(x, shape, strides, padding)

    @staticmethod
    def conv2d(x, shape, strides, padding):
        return tf.nn.conv2d(x, Model.weight(shape), strides, padding) + Model.bias(shape[-1:])

    def set_inputs(inputs):
        self.inputs = inputs

    def set_outputs(outputs):
        self.outputs = outputs

    def set_init()
        # sess
