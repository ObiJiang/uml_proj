import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from misc import AttrDict, sample_floats
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class MetaCluster():
    def __init__(self,config):
        self.config = config
        self.n_unints = 32
        self.batch_size = config.batch_size
        self.k = 2
        self.num_sequence = 30
        self.lr = 0.01
        self.model = self.model()
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
        vars_ = {var.name.split(":")[0]: var for var in vars}
        self.saver = tf.train.Saver(vars_,max_to_keep=config.max_to_keep)

    def create_dataset(self):
        xcenters = sample_floats(-1,1,2)
        ycenters = sample_floats(-1,1,2)

        #labels = np.random.randint(2, size=self.num_sequence)
        labels = np.arange(self.num_sequence)%2
        np.random.shuffle(labels)

        data = np.zeros((self.num_sequence,2))

        mean = (xcenters[0],ycenters[0])
        cov = [[0.01, 0], [0, 0.01]]
        data[labels==1,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==1)))

        mean = (xcenters[1],ycenters[1])
        data[labels==0,:] = np.random.multivariate_normal(mean, cov, (np.sum(labels==0)))
        if self.config.show_graph:
            plt.scatter(data[labels==1,0], data[labels==1,1])
            plt.scatter(data[labels==0,0], data[labels==0,1])
            plt.show()

        return np.expand_dims(data,axis=0),np.expand_dims(labels,axis=0).astype(np.int32)

    def model(self):
        sequences = tf.placeholder(tf.float32, [self.batch_size,None, 2])
        labels = tf.placeholder(tf.int32, [self.batch_size,None])


        sequences_tile = tf.tile(sequences,[1,5,1])
        labels_tile = tf.tile(labels,[1,5])
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
        with tf.variable_scope('core'):
            output, states = tf.nn.dynamic_rnn(cell, sequences_tile, dtype=tf.float32, initial_state = cell_init_state)

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
        with tf.variable_scope('core'):
            policy = tf.layers.dense(output,self.k)

        """ Define Loss and Optimizer """
        loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_tile[:,120:] ,logits= policy[:,120:])),
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.mod(labels_tile[:,120:]+1,2) ,logits= policy[:,120:]))]

        miss_list_0 = tf.not_equal(tf.cast(tf.argmax(policy,axis=2),tf.float64)[:,120:],tf.cast(labels,tf.float64))
        miss_list_1 = tf.not_equal(tf.cast(tf.argmax(policy,axis=2),tf.float64)[:,120:],tf.cast(tf.mod(labels+1,2),tf.float64))

        miss_rate_0 = tf.reduce_sum(tf.cast(miss_list_0,tf.float32))/(self.num_sequence*self.batch_size)
        miss_rate_1 = tf.reduce_sum(tf.cast(miss_list_1,tf.float32))/(self.num_sequence*self.batch_size)

        miss_rate = tf.minimum(miss_rate_0,miss_rate_1)

        # l2 = 0.0001 * sum(
        #     tf.nn.l2_loss(tf_var)
        #         for tf_var in tf.trainable_variables()
        #         if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        # )

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss[0])
        return AttrDict(locals())

    def train(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(1):
            #_,_,miss_rate = sess.run([model.keep_state_op,model.opt,model.miss_rate],feed_dict={model.sequences:data,model.labels:labels})
            _,miss_rate = sess.run([model.opt,model.miss_rate],feed_dict={model.sequences:data,model.labels:labels})
            if epoch_ind == 0:
                print("Epochs{}:{}".format(epoch_ind,miss_rate))
        print("Epochs{}:{}".format(epoch_ind,miss_rate))

    def test(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(100):
            states,miss_rate,loss = sess.run([model.keep_state_op,model.miss_rate,model.loss],feed_dict={model.sequences:data,model.labels:labels})
            print("Epochs{}:{}".format(epoch_ind,miss_rate))

    def save_model(self, sess, epoch):
        print('\nsaving model...')

        # create path if not around
        model_save_path = self.config.model_save_dir
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

        model_name = '{}/model'.format(model_save_path)
        save_path = self.saver.save(sess, model_name, global_step = epoch)
        print('model saved at', save_path, '\n\n')

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--show_graph', default=False, action='store_true')
    parser.add_argument('--max_to_keep', default=3, type=int)
    parser.add_argument('--model_save_dir', default='./out')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--training_exp_num', default=1000, type=int)

    config = parser.parse_args()

    if not config.test:
        metaCluster = MetaCluster(config)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # training
            for _ in tqdm(range(config.training_exp_num)):
                data_list = []
                labels_list = []
                for _ in range(config.batch_size):
                    data_one, labels_one = metaCluster.create_dataset()
                    data_list.append(data_one)
                    labels_list.append(labels_one)
                data = np.concatenate(data_list)
                labels = np.concatenate(labels_list)
                metaCluster.train(data,labels,sess)

            # saving models ...
            metaCluster.save_model(sess,config.training_exp_num)

            # testing
            data_list = []
            labels_list = []
            for _ in range(config.batch_size):
                data_one, labels_one = metaCluster.create_dataset()
                data_list.append(data_one)
                labels_list.append(labels_one)
            data = np.concatenate(data_list)
            labels = np.concatenate(labels_list)
            metaCluster.test(data,labels,sess)
    else:
        config.batch_size = 1
        metaCluster = MetaCluster(config)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
            vars_ = {var.name.split(":")[0]: var for var in vars}
            saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)
            save_dir = config.model_save_dir

            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
            print("Loading saved model from {}".format(save_path))
            saver.restore(sess, save_path)

            data, labels = metaCluster.create_dataset()
            metaCluster.test(data,labels,sess)

            # labels = (labels+1)%2
            # metaCluster.test(data,labels,sess)
