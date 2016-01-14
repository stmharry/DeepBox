import tensorflow as tf
import deepbox as db

from scipy.io import loadmat
from numpy import concatenate
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test)) = mnist.load_data()

dataset = db.Dataset(
    data=[concatenate([X_train, X_test])], concatenate([y_train, y_test])[None, :]], 
    length=[y_train.size, 0, y_test.size],
    normalize=True, one_hot=True)

#
model = db.Model()

x = tf.placeholder('float', shape=(None,) + dataset.input_dim)
y = tf.placeholder('float', shape=(None,) + dataset.output_dim)
d = tf.placeholder('float')

h = tf.reshape(x, (-1,) + dataset.dim + (1,)) 
h = db.Model.conv2d(h, (5, 5, 1, 32), (1, 1, 1, 1), 'SAME')
h = tf.nn.relu(h)
h = db.Model.max_pool(h, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

h = db.Model.conv2d(h, (5, 5, 32, 64), (1, 1, 1, 1), 'SAME')
h = tf.nn.relu(h)
h = db.Model.max_pool(h, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

h = tf.reshape(h, (-1,) + db.Model.get_size(h))
h = db.Model.dense(h, (db.Model.get_size(h), 1024))
h = tf.nn.relu(h)
h = tf.nn.dropout(h, d)

h = db.Model.dense(h, (1024, 10))
yp = tf.nn.softmax(h)

ce = -tf.reduce_sum(y * tf.log(yp))
train = tf.train.AdamOptimizer(1e-3).minimize(ce)

model.set_inputs([x, d])
model.set_outputs([y])

model.set_init()
model.set_train()
model.save()
