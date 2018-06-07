# coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
import time

######### PARAMETERS ########

time_step = 60  # 时间步
batch_size = 1000  # 每一批次训练多少个样例
INPUT_SIZE = 9  # 输入层维度
OUTPUT_SIZE = 3  # 输出层维度
CELL_SIZE = 128  # hidden unit size
lr = 0.001  # 学习率
layers = 3
WEIGHT = 0.3

predict_day = int(10)
classes = 3


train_x = np.random.uniform(-1, 1, [500000, time_step, INPUT_SIZE])
train_y = np.random.randint(0, 1, [500000, OUTPUT_SIZE])


######## WEIGHT & BIAS ########
weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, CELL_SIZE])),
    'out': tf.Variable(tf.random_normal([CELL_SIZE, OUTPUT_SIZE]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[CELL_SIZE], )),
    'out': tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE, ]))}

######## INPUTS ########

X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

batch_index = []
for i in range(len(train_y)):
    if i % batch_size == 0:
        batch_index.append(i)
keep_prob = tf.placeholder(tf.float32)


######## DEFINE LSTM #######


def lstm(X):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, CELL_SIZE])  # 将tensor转成3维，作为lstm cell的输入
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(CELL_SIZE, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(layers)], state_is_tuple=True)
    init_state = mcell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(mcell, input_rnn, initial_state=init_state, time_major=False)
    h_state = final_states[-1][1]
    w_out = weights['out']
    b_out = biases['out']
    results = tf.nn.softmax(tf.matmul(h_state, w_out) + b_out)
    return results


def train_lstm():
    pred = lstm(X)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
    cost = tf.reduce_mean(weight_entropy(logits=pred, labels=Y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # 重复训练2000次
        for i in range(150):
            for step in range(len(batch_index) - 1):
                _, acc_ = sess.run([train_op, accuracy], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                    Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                    keep_prob: 0.8})
                # print("Iter%d, step %d, training accuracy %f" % (i,step, acc_,))
            if i % 5 == 0:
                # print("保存模型：",saver.stock(sess,'test.model',global_step=i))
                train_accuracy, prediction, label = sess.run([accuracy, pred, Y], feed_dict={
                    X: train_x[batch_index[step]:batch_index[step + 1]],
                    Y: train_y[batch_index[step]:batch_index[step + 1]], keep_prob: 1.0})
                print("Iter,%f" % train_accuracy)
                print('precision_P,%f' % count_precision_P(np.argmax(prediction, 1), np.argmax(label, 1)))
                print('precision_N,%f' % count_precision_N(np.argmax(prediction, 1), np.argmax(label, 1)))

    # print("test accuracy %f" % sess.run(accuracy, feed_dict={X: train_x, Y: train_y, keep_prob: 1.0}))


def count_precision_P(pre, real):
    TP, FP, length = 0, 0, len(pre)
    for i in range(length):
        if pre[i] == 2 and real[i] == 2:
            TP += 1
        elif pre[i] == 2 and real[i] != 2:
            FP += 1
    try:
        return TP / (TP + FP)
    except:
        return .0


def count_precision_N(pre, real):
    TN, FN, length = 0, 0, len(pre)
    for i in range(length):
        if pre[i] == 0 and real[i] == 0:
            TN += 1
        elif pre[i] == 0 and real[i] != 0:
            FN += 1
    try:
        return TN / (TN + FN)
    except:
        return .0


def weight_entropy(logits, labels):
    if True:
        weight = WEIGHT
    else:
        weight = 1
    return weight * (-tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0))))


if __name__ == '__main__':
    t3 = time.time()
    train_lstm()
    t4 = time.time()
    print('train use: %f' % (t4 - t3))
