import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from misc import AttrDict, sample_floats
import matplotlib.pyplot as plt

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class MetaCluster():
    def __init__(self):
        self.n_unints = 32
        self.batch_size = 1
        self.k = 2
        self.num_sequence = 100
        self.lr = 0.01
        self.model = self.model()

    def create_dataset(self):
        xcenters = sample_floats(1,10,2)
        ycenters = sample_floats(1,10,2)

        labels = np.random.randint(2, size=self.num_sequence)
        data = np.zeros((self.num_sequence,2))

        mean = (xcenters[0],ycenters[0])
        cov = [[0.1, 0], [0, 0.1]]
        data[labels==1,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==1)))

        mean = (xcenters[1],ycenters[1])
        data[labels==0,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==0)))

        # plt.scatter(data[labels==1,0], data[labels==1,1])
        # plt.scatter(data[labels==0,0], data[labels==0,1])
        # plt.show()

        return np.expand_dims(data,axis=0),np.expand_dims(labels,axis=0).astype(np.int32)

    def model(self):
        sequences = tf.placeholder(tf.float32, [self.batch_size,None, 2])
        labels = tf.placeholder(tf.int32, [self.batch_size,None])

        # cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_unints,state_is_tuple=True)
        cells = [tf.contrib.rnn.BasicLSTMCell(32) for _ in range(2)]
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        """ Save init states (zeros) """
        with tf.variable_scope('Hidden_states'):
            state_variables = []
            for s_c, s_h in cell.zero_state(self.batch_size,tf.float32):
                state_variables.append(
                        tf.nn.rnn_cell.LSTMStateTuple(
                        tf.Variable(s_c,trainable=False),
                        tf.Variable(s_h,trainable=False))
                    )

        cell_init_state = tuple(state_variables)

        """ Define LSTM network """
        with tf.variable_scope('LSTM'):
            output, states = tf.nn.dynamic_rnn(cell, sequences, dtype=tf.float32)#, initial_state = cell_init_state)

        """ Keep and Clear Op """
        # keep state op
        update_ops = []
        for state_variables, state in zip(cell_init_state, states):
            update_ops.extend([ state_variables[0].assign(state[0]),
                                state_variables[1].assign(state[1])])
        keep_state_op = tf.tuple(update_ops)

        # clear state op
        update_ops = []
        for state_variables, state in zip(cell_init_state, states):
            update_ops.extend([ state_variables[0].assign(tf.zeros_like(state[0])),
                                state_variables[1].assign(tf.zeros_like(state[1]))])
        clear_state_op = tf.tuple(update_ops)

        """ Define Policy and Value """
        # policy = slim.fully_connected(output, self.k,
        #         activation_fn=None,
        #         weights_initializer=normalized_columns_initializer(0.01),
        #         biases_initializer=None)
        # value = slim.fully_connected(output, 1,
        #         activation_fn=None,
        #         weights_initializer=normalized_columns_initializer(1.0),
        #         biases_initializer=None)

        policy = tf.layers.dense(output,self.k)

        """ Define Loss and Optimizer """
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels ,logits= policy))
        miss_list = tf.not_equal(tf.cast(tf.argmax(policy,axis=2),tf.float64),tf.cast(labels,tf.float64))
        miss_rate = tf.reduce_sum(tf.cast(miss_list,tf.float32))/self.num_sequence

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return AttrDict(locals())

    def train(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(100):
            _,_,loss = sess.run([model.keep_state_op,model.opt,model.miss_rate],feed_dict={model.sequences:data,model.labels:labels})
        print("Epochs{}:{}".format(epoch_ind,loss))

    def test(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(100):
            _,loss = sess.run([model.keep_state_op,model.loss],feed_dict={model.sequences:data,model.labels:labels})
            print("Epochs{}:{}".format(epoch_ind,loss))

metaCluster = MetaCluster()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for _ in range(100):
        data, labels = metaCluster.create_dataset()
        metaCluster.train(data,labels,sess)
    # testing
    data, labels = metaCluster.create_dataset()
    metaCluster.test(data,labels,sess)
