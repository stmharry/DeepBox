import numpy as np
import tensorflow as tf
import scipy.io

import deepbox as db

_DATA_DIR = './data/'
_SEL = 3

_FEAT = 'surf'
_DOMAIN = 'webcam'
_ARCH = [1024]  # 1000-1024-10
_LEARN_RATE = 1e-3
_KWARGS = {'epoch': 16, 'batch_size': -1, 'report_per': 1}


def normalize(X):
    _X = np.concatenate(X)
    m = np.max(_X, 0) + 1e-7
    return [x / m for x in X]


def get(feat, domain):
    np.random.seed(1337)

    mat = scipy.io.loadmat(_DATA_DIR + feat + '_' + domain + '.mat')
    (X, Y) = (mat['X'], mat['Y'][:, 0])
    X = db.Dataset.normalize([X])[0]
    Y = db.Dataset.one_hot([Y])[0][0]
    (N, DX) = X.shape
    (N, DY) = Y.shape
    (U, L) = [x.T.flatten() for x in np.split(np.argsort(Y * np.random.random_sample((N, 1)), 0), [-_SEL])]
    return (X, Y, N, DX, DY, U, L)


(X, Y, N, DX, DY, U, L) = get(_FEAT, _DOMAIN)

x = tf.placeholder('float', (None, DX))
y = tf.placeholder('float', (None, DY))
h = db.Model.dense_sequential(x, 1.0, _ARCH)
yp = tf.nn.softmax(db.Model.dense(h, DY))

loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(yp), 1), 0)
train = tf.train.AdamOptimizer(_LEARN_RATE).minimize(loss)

correct = tf.equal(tf.argmax(y, 1), tf.argmax(yp, 1))
acc = tf.reduce_mean(tf.to_float(correct))

model = db.Model()
model.init()
model.set_train({x: X[L], y: Y[L]}, {loss: 'loss', acc: 'acc'}, {train: None}, **_KWARGS)
model.set_val({x: X[U], y: Y[U]}, {loss: 'loss', acc: 'acc'})
model.train()
