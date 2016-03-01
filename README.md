# DeepBox 
---

A useful toolbox on top of TensorFlow.

## Requirements
- Python ≥ 2.7.10
- TensorFlow ≥ 0.6.1
- NumPy ≥ 1.10.4
- SciPy ≥ 0.16.1

## Features
### Light-weighted wrapper for TensorFlow
```python
import deepbox as db

x = tf.placeholder('float', (None, 1000))
y = tf.placeholder('float', (None, 10))
h = db.Model.dense_sequential(x, 1.0, [1024])  # 1000-1024-10
yp = tf.nn.softmax(db.Model.dense(h, 10))

loss = - tf.reduce_mean(y * tf.log(yp))
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

model = db.Model()
model.init()
model.set_train({x: X, y: Y}, {loss: 'loss'}, {train: None}, epoch=16)
model.train()
```
### Auto mini-batch training with shuffling
### Nice progress bar
```
Epoch:  1/16 [====================]  train-loss: 7.9826
Epoch:  2/16 [====================]  train-loss: 3.1198
...
Epoch: 15/16 [====================]  train-loss: 0.0009
Epoch: 16/16 [==========----------]  train-loss: 0.0007
```
### Validation friendly
```python
outputs = {loss: 'loss'}
updates = {train: None}
model.set_train({x: XT, y: YT}, outputs, updates)
model.set_val({x: XV, y: YV}, outputs)
model.train()
```

There are many more features but due to limited brain power I will update them in the future. Please explore more in `main.py` and `deepbox.py`.