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


# Flags for defining the tf.train.ClusterSpec
flags = tf.app.flags
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

######## DEFINE LSTM #######


def main(unused_argv):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        is_chief = (FLAGS.task_index == 0)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.contrib.framework.get_or_create_global_step()

            pred = lstm(X)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
            cost = tf.reduce_mean(weight_entropy(logits=pred, labels=Y))
            train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            init_op = tf.global_variables_initializer()
            summary_op = tf.summary.merge_all()

            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
            # module_file = tf.train.latest_checkpoint()

        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % FLAGS.task_index])

        hooks = [tf.train.StopAtStepHook(last_step=100)]
        with tf.train.MonitoredTrainingSession(server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks,
                                               save_checkpoint_secs=60,
                                               config=sess_config) as sess:
            time_begin = time.time()
            print("Training begins @ %f" % time_begin)
            local_step = 0
            while not sess.should_stop():
                _, step, acc_ = sess.run([train_op, global_step, accuracy],
                                         feed_dict={X: train_x[batch_index[local_step]:batch_index[local_step + 1]],
                                                    Y: train_y[batch_index[local_step]:batch_index[local_step + 1]],
                                                    keep_prob: 0.8})
                print("local_step %d (global step: %d), training accuracy %f" % (local_step, step, acc_,))

                if local_step % 10 == 0:
                    train_accuracy, prediction, label = sess.run([accuracy, pred, Y], feed_dict={
                        X: train_x[batch_index[local_step]:batch_index[local_step + 1]],
                        Y: train_y[batch_index[local_step]:batch_index[local_step + 1]], keep_prob: 1.0})
                    print("Iter,%f" % train_accuracy)
                    print('precision_P,%f' % count_precision_P(np.argmax(prediction, 1), np.argmax(label, 1)))
                    print('precision_N,%f' % count_precision_N(np.argmax(prediction, 1), np.argmax(label, 1)))

                local_step += 1
            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)


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
    tf.app.run()
