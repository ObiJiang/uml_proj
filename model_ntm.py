import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from misc import AttrDict, sample_floats
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from NeuralTuringMachine.ntm import NTMCell,NTMControllerState
from mnist import Generator_minst

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
        self.num_sequence = 100
        self.lr = 0.001
        self.fea = 2
        self.model = self.model()
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
        vars_ = {var.name.split(":")[0]: var for var in vars}
        self.saver = tf.train.Saver(vars_,max_to_keep=config.max_to_keep)

    def create_dataset(self):
        labels = np.arange(self.num_sequence)%self.k
        np.random.shuffle(labels)

        data = np.zeros((self.num_sequence,self.fea))

        mean = np.random.rand(self.k, self.fea)*2-1

        sort_ind = np.argsort(mean[:,0])

        for label_ind,ind in enumerate(sort_ind):
            s = np.random.uniform(0.1,0.01,self.fea)
            cov = np.diag(s)
            data[labels==label_ind,:] = np.random.multivariate_normal(mean[ind, :], cov, (np.sum(labels==label_ind)))
        if self.config.show_graph:
            for i in range(self.k):
                plt.scatter(data[labels==i,0], data[labels==i,1])
            plt.show()

        return np.expand_dims(data,axis=0),np.expand_dims(labels,axis=0).astype(np.int32)

    def model(self):
        sequences = tf.placeholder(tf.float32, [self.batch_size,None, 2])
        labels = tf.placeholder(tf.int32, [self.batch_size,None])

        """ Define Neural Turing Machine """
        num_controller_layers = 2
        num_controller_units = 16
        num_memory_locations = 8
        memory_size = 16
        num_read_heads = 8
        num_write_heads = 8
        num_bits_per_output_vector = 32
        clip_controller_output_to_value = 3
        cell = NTMCell(num_controller_layers, num_controller_units, num_memory_locations, memory_size,
            num_read_heads, num_write_heads, shift_range=3, output_dim=num_bits_per_output_vector,
            clip_value=clip_controller_output_to_value)

        with tf.variable_scope('Hidden_states'):
            cell_zero_state = cell.zero_state(self.batch_size,tf.float32)
            state_variables = []
            for s_c, s_h in cell_zero_state.controller_state:
                state_variables.append(
                        tf.nn.rnn_cell.LSTMStateTuple(
                        tf.Variable(s_c,trainable=False),
                        tf.Variable(s_h,trainable=False))
                    )
            controller_state_init = tuple(state_variables)

            state_variables = []
            for read_vector in cell_zero_state.read_vector_list:
                state_variables.append(
                        tf.Variable(read_vector,trainable=False)
                    )
            read_vector_list_init = state_variables

            M_init = tf.Variable(cell_zero_state.M,trainable=False)

            state_variables = []
            for w in cell_zero_state.w_list:
                state_variables.append(
                        tf.Variable(w,trainable=False)
                    )
            w_list_init = state_variables

        cell_init_state = NTMControllerState(
            controller_state=controller_state_init,
            read_vector_list=read_vector_list_init,
            w_list=w_list_init,
            M=M_init)

        """ Define NTM network """
        with tf.variable_scope('core'):
            output, states = tf.nn.dynamic_rnn(cell, sequences, dtype=tf.float32, initial_state = cell_init_state)

        """ Keep and Clear Op """

        # clear state op
        update_ops = []
        for state_variables, state in zip(cell_init_state.controller_state, states.controller_state):
            update_ops.extend([state_variables[0].assign(state[0]),
                                state_variables[1].assign(state[1])])

        for state_variables, state in zip(cell_init_state.read_vector_list, states.read_vector_list):
            update_ops.extend([state_variables.assign(state)])

        update_ops.extend([cell_init_state.M.assign(states.M)])

        for state_variables, state in zip(cell_init_state.w_list, states.w_list):
            update_ops.extend([state_variables.assign(state)])

        keep_state_op = tf.tuple(update_ops)

        # clear state op
        update_ops = []
        for state_variables, state in zip(cell_init_state.controller_state, cell_zero_state.controller_state):
            update_ops.extend([state_variables[0].assign(state[0]),
                                state_variables[1].assign(state[1])])

        for state_variables, state in zip(cell_init_state.read_vector_list, cell_zero_state.read_vector_list):
            update_ops.extend([state_variables.assign(state)])

        update_ops.extend([cell_init_state.M.assign(cell_zero_state.M)])

        for state_variables, state in zip(cell_init_state.w_list, cell_zero_state.w_list):
            update_ops.extend([state_variables.assign(state)])

        clear_state_op = tf.tuple(update_ops)


        """ Define Policy and Value """
        with tf.variable_scope('core'):
            policy = tf.layers.dense(output,self.k)

        """ Define Loss and Optimizer """
        loss_batch_class = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels ,logits= policy),axis=1),
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.mod(labels+1,2) ,logits= policy),axis=1)]


        loss_batch = tf.minimum(loss_batch_class[0],loss_batch_class[1])

        loss = tf.reduce_mean(loss_batch)

        miss_list_0 = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(tf.argmax(policy,axis=2),tf.float64),tf.cast(labels,tf.float64)),tf.float32),axis=1)
        miss_list_1 = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(tf.argmax(policy,axis=2),tf.float64),tf.cast(tf.mod(labels+1,2),tf.float64)),tf.float32),axis=1)

        miss_rate = tf.reduce_sum(tf.minimum(miss_list_0,miss_list_1),axis=0)/(self.num_sequence*self.batch_size)

        predicted_label = tf.argmax(policy,axis=2)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return AttrDict(locals())

    def train(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(30):
            _,_,miss_rate = sess.run([model.keep_state_op,model.opt,model.miss_rate],feed_dict={model.sequences:data,model.labels:labels})
        print("Epochs{}:{}".format(epoch_ind,miss_rate))

    def test(self,data,labels,sess,validation=False):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(30):
            _,miss_rate,loss = sess.run([model.keep_state_op,model.miss_rate,model.loss],feed_dict={model.sequences:data,model.labels:labels})
            if not validation:
                print("Epochs{}:{}".format(epoch_ind,miss_rate))
        if validation:
            print("Epochs{}:{}".format(epoch_ind,miss_rate))
        if config.show_comparison_graph:
            data = np.squeeze(data)
            labels = np.squeeze(labels)
            predicted_label = np.squeeze(predicted_label)
            diff = np.abs(labels-predicted_label)

            fig = plt.figure()
            ax = fig.add_subplot(311)

            for i in range(self.k):
                ax.scatter(data[labels==i,0], data[labels==i,1])
            ax.set_title('Original',fontsize=8)
            #ax.axis('scaled')

            ax = fig.add_subplot(312)
            for i in range(self.k):
                ax.scatter(data[predicted_label==i,0], data[predicted_label==i,1])
            ax.set_title('Predicton',fontsize=8)
            #ax.axis('scaled')

            ax = fig.add_subplot(313)
            ax.scatter(data[diff==0,0], data[diff==0,1],color='black')
            ax.scatter(data[diff==1,0], data[diff==1,1],color='red')
            ax.set_title('Difference',fontsize=8)
            #ax.axis('scaled')

            plt.show()

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
    parser.add_argument('--show_comparison_graph', default=False, action='store_true')
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
            for train_ind in tqdm(range(config.training_exp_num)):
                data_list = []
                labels_list = []
                for _ in range(config.batch_size):
                    data_one, labels_one = metaCluster.create_dataset()
                    data_list.append(data_one)
                    labels_list.append(labels_one)
                data = np.concatenate(data_list)
                labels = np.concatenate(labels_list)
                metaCluster.train(data,labels,sess)

                if train_ind % 10 == 0:
                    print('-----validation-----')
                    # validation
                    data_list = []
                    labels_list = []
                    for _ in range(config.batch_size):
                        data_one, labels_one = metaCluster.create_dataset()
                        data_list.append(data_one)
                        labels_list.append(labels_one)
                    data = np.concatenate(data_list)
                    labels = np.concatenate(labels_list)
                    metaCluster.test(data,labels,sess,validation=True)

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

            generator = Generator_minst(metaCluster.fea)
            data, labels = generator.generate(metaCluster.num_sequence//2)
            data = np.expand_dims(data, axis=0)
            labels = np.expand_dims(labels, axis=0)
            #data, labels = metaCluster.create_dataset()
            metaCluster.test(data,labels,sess)
