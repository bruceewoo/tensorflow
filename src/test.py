import tensorflow as tf
import numpy as np
import time
import os
batch_size = 256
data = np.random.random([512000,10,30])
xx = tf.placeholder(tf.float32,[batch_size,10,30])
def make_cell():
    return tf.contrib.rnn.LSTMCell(256, forget_bias=0.0)
cell=tf.contrib.rnn.MultiRNNCell([make_cell() for x in range(2)], state_is_tuple=True)
outputs, last_state = tf.nn.dynamic_rnn(cell, xx, dtype=tf.float32,scope='pi_lstm')
logits = tf.reduce_mean(outputs[0])
sess =tf.Session()
sess.run(tf.global_variables_initializer())
pre_time = time.time()
for i in range(10):
    for idx in range(0,data.shape[0],batch_size):
        sess.run(logits,feed_dict={xx:data[idx:idx+batch_size]})
    dur = time.time() - pre_time
    print('%s %ss' % (idx, dur))
    pre_time = time.time()
