import tensorflow as tf
import deepbox as db
import numpy as np
import os

from scipy.io import loadmat
from keras.datasets import mnist

def get_path(issave):
    isload = not issave
    root = 'temp/'
    i = 0
    while True:
        path = root + str(i) + '/model'
        if os.path.isfile(path):
            i += 1
            if isload:
                yield path
        else:
            directory = root + str(i)
            if issave and not os.path.isdir(directory):
                os.mkdir(directory)
            break
    yield path

def get_save_path():
    return [p for p in get_path(True)][0]

def get_load_path():
    for p in get_path(False):
        yield p

if 'dataset' not in locals():
    ((X_train, y_train), (X_test, y_test)) = mnist.load_data()

    X_train = X_train[0:1000]
    y_train = y_train[0:1000]
    X_test = X_test[0:1000]
    y_test = y_test[0:1000]

    dev_ratio = 0.8
    len_dev = int(dev_ratio * X_train.shape[0])
    
    (X_dev, X_val) = np.split(X_train, [len_dev])
    (y_dev, y_val) = np.split(y_train, [len_dev])

    (X_dev, X_val, X_test) = db.Dataset.normalize([X_dev, X_val, X_test])
    ((y_dev, y_val, y_test), y0) = db.Dataset.one_hot([y_dev, y_val, y_test])

if 'model' not in locals():
    x = tf.placeholder('float', shape=[None, 28, 28])
    y = tf.placeholder('float', shape=[None, 10])
    d = tf.placeholder('float')

    h = tf.reshape(x, (-1, 28, 28, 1)) 
    h = db.Model.conv2d(h, 32, (5, 5), (1, 1), 'SAME')
    h = tf.nn.relu6(h)
    h = db.Model.max_pool(h, (2, 2), (2, 2), 'SAME')

    h = db.Model.conv2d(h, 64, (5, 5), (1, 1), 'SAME')
    h = tf.nn.relu6(h)
    h = db.Model.max_pool(h, (2, 2), (2, 2), 'SAME')

    h = tf.reshape(h, [-1, db.Model.get_size(h)])
    h = db.Model.dense(h, 1024)
    h = tf.nn.relu6(h)
    h = tf.nn.dropout(h, d)
    h = db.Model.dense(h, 10)
    yp = tf.nn.softmax(h)

    loss = -tf.reduce_mean(y * tf.log(yp))
    correct = tf.equal(tf.argmax(yp, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))

    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

model = db.Model()
model.init()

model.set(db.TRAIN, {x:X_dev, y:y_dev, d:0.5}, {loss:'loss', acc:'acc'}, {train:None}, epoch=1, batch_size=128)
model.set(db.VAL, {x:X_val, y:y_val, d:0.5}, {loss:'loss', acc:'acc'})
model.set(db.TEST, {x:X_test, y:y_test, d:1.0}, {yp:None})

