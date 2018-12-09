import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from misc import AttrDict, sample_floats
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from tensorflow.python.ops.rnn import _transpose_batch_time
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from edu import eduGenerate     # seq=100 fea=5
from mnist import Generator_minst

from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# attention + bi-directional
# maml
# put lstm ouput into lstm
# reptile + ntm
# just 5 iterations
# also try 10 clusters
# try agent and rl update rule
# shuffle
# mimic k-means loss
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
        self.k = config.k
        self.num_sequence = 150
        self.fea = config.fea
        self.lr = 0.003
        self.keep_prob = 0.8
        self.alpha = 0.2
        self.model = self.model()
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='core')
        vars_ = {var.name.split(":")[0]: var for var in vars}
        self.saver = tf.train.Saver(vars_,max_to_keep=config.max_to_keep)

    def create_dataset(self):
        labels = np.arange(self.num_sequence)%self.k
        np.random.shuffle(labels)

        data = np.zeros((self.num_sequence,self.fea))

        mean = np.random.rand(self.k, self.fea)*2-1

        #cov = np.identity(self.fea)*0.1

        sort_ind = np.argsort(mean[:,0])

        for label_ind,ind in enumerate(sort_ind):
            cov_factor = np.random.rand(1)*10+10
            cov = np.random.normal(size=(self.fea,self.fea))/np.sqrt(self.fea*cov_factor)
            cov = cov.T @ cov
            # cov = np.random.normal(size=(self.fea,self.fea))/np.sqrt(self.fea*100)
            # cov = cov.T @ cov
            data[labels==label_ind,:] = np.random.multivariate_normal(mean[ind, :], cov, (np.sum(labels==label_ind)))
        if self.config.show_graph:
            for i in range(self.k):
                plt.scatter(data[labels==i,0], data[labels==i,1])
                print(i)
            plt.show()

        return np.expand_dims(data,axis=0),np.expand_dims(labels,axis=0).astype(np.int32)

    def sampling_rnn(self, cell, initial_state, input_, seq_lengths):

        # raw_rnn expects time major inputs as TensorArrays
        max_time = seq_lengths+1  # this is the max time step per batch
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, clear_after_read=False)
        inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # model_input is the input placeholder
        output_dim = self.k  # the dimensionality of the model's output at each time step
        input_dim = input_.get_shape()[-1].value +  output_dim # the dimensionality of the input to each time step

        def loop_fn(time, cell_output, cell_state, loop_state):
            """
            Loop function that allows to control input to the rnn cell and manipulate cell outputs.
            :param time: current time step
            :param cell_output: output from previous time step or None if time == 0
            :param cell_state: cell state from previous time step
            :param loop_state: custom loop state to share information between different iterations of this loop fn
            :return: tuple consisting of
              elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
                needed because of variable sequence size
              next_input: input to next time step
              next_cell_state: cell state forwarded to next time step
              emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
                but could e.g. be the output of a dense layer attached to the rnn layer.
              next_loop_state: loop state forwarded to the next time step
            """
            if cell_output is None:
                # time == 0, used for initialization before first call to cell
                next_cell_state = initial_state
                # the emit_output in this case tells TF how future emits look
                emit_output = tf.zeros([output_dim])
            else:
                # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                # here you can do whatever ou want with cell_output before assigning it to emit_output.
                # In this case, we don't do anything
                next_cell_state = cell_state
                emit_output = cell_output

            # check which elements are finished
            elements_finished = (time >= seq_lengths)
            finished = tf.reduce_all(elements_finished)

            # assemble cell input for upcoming time step
            current_output = emit_output if cell_output is not None else None
            input_original = inputs_ta.read(time)  # tensor of shape (None, input_dim)

            if current_output is None:
                # this is the initial step, i.e. there is no output from a previous time step, what we feed here
                # can highly depend on the data. In this case we just assign the actual input in the first time step.
                next_in = tf.concat([input_original, tf.zeros([self.batch_size,output_dim])],axis=1)
            else:
                # time > 0, so just use previous output as next input
                # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
                # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
                next_in = tf.concat([input_original,current_output],axis=1)

            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, input_dim], dtype=tf.float32),  # copy through zeros
                                 lambda: next_in)  # if not finished, feed the previous output as next input

            # set shape manually, otherwise it is not defined for the last dimensions
            next_input.set_shape([None, input_dim])

            # loop state not used in this example
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = _transpose_batch_time(outputs_ta.stack())
        final_state = last_state

        return outputs, final_state

    def model(self):
        sequences = tf.placeholder(tf.float32, [self.batch_size,None, self.fea])
        labels = tf.placeholder(tf.int32, [self.batch_size,None])

        # cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_unints,state_is_tuple=True)
        cells = [tf.contrib.rnn.BasicLSTMCell(n_unint) for n_unint in [32,32,self.k]]
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
            #output, states = tf.nn.dynamic_rnn(cell, sequences, dtype=tf.float32, initial_state = cell_init_state)
            output, states = self.sampling_rnn(cell, cell_init_state,sequences, self.num_sequence)
            #output, states = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell, sequences, dtype=tf.float32, initial_states_fw = cell_init_state)

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
            # atten_weights = tf.matmul(output,output,transpose_b=True)
            # attended_output = tf.reduce_sum(tf.expand_dims(atten_weights,axis=3)*tf.expand_dims(output,axis=2),axis=2)
            # policy = tf.layers.dense(attended_output,self.k)
            # policy = tf.layers.dense(output,self.k)
            policy = output



    # def model(self):
    #     sequences = tf.placeholder(tf.float32, [self.batch_size,None, self.fea])
    #     labels = tf.placeholder(tf.int32, [self.batch_size,None])
    #
    #     # cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_unints,state_is_tuple=True)
    #     fw_cells = [tf.contrib.rnn.BasicLSTMCell(n_unints) for n_unints in [32,32]]
    #     fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
    #
    #
    #     bw_cells = [tf.contrib.rnn.BasicLSTMCell(n_unints) for n_unints in [32,32]]
    #     bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
    #
    #     """ Save init states (zeros) """
    #     with tf.variable_scope('FW_Hidden_states'):
    #         state_variables = []
    #         for s_c, s_h in fw_cell.zero_state(self.batch_size,tf.float32):
    #             state_variables.append(
    #                     tf.nn.rnn_cell.LSTMStateTuple(
    #                     tf.Variable(s_c,trainable=False),
    #                     tf.Variable(s_h,trainable=False))
    #                 )
    #
    #         cell_init_state_fw = tuple(state_variables)
    #
    #
    #     with tf.variable_scope('BW_Hidden_states'):
    #         state_variables = []
    #         for s_c, s_h in bw_cell.zero_state(self.batch_size,tf.float32):
    #             state_variables.append(
    #                     tf.nn.rnn_cell.LSTMStateTuple(
    #                     tf.Variable(s_c,trainable=False),
    #                     tf.Variable(s_h,trainable=False))
    #                 )
    #
    #         cell_init_state_bw = tuple(state_variables)
    #
    #     """ Define LSTM network """
    #     with tf.variable_scope('core'):
    #         #output, states = tf.nn.dynamic_rnn(cell, sequences, dtype=tf.float32, initial_state = cell_init_state)
    #         output, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sequences, dtype=tf.float32, initial_state_fw = cell_init_state_fw,  initial_state_bw = cell_init_state_bw)
    #
    #     """ Keep and Clear Op """
    #     # keep state op
    #     update_ops = []
    #     for state_variables, state in zip(cell_init_state_fw, states[0]):
    #         update_ops.extend([ state_variables[0].assign(state[0]),
    #                             state_variables[1].assign(state[1])])
    #
    #     for state_variables, state in zip(cell_init_state_bw, states[1]):
    #         update_ops.extend([ state_variables[0].assign(state[0]),
    #                             state_variables[1].assign(state[1])])
    #
    #     keep_state_op = tf.tuple(update_ops)
    #
    #     # clear state op
    #     update_ops = []
    #     for state_variables, state in zip(cell_init_state_fw, states[0]):
    #         update_ops.extend([ state_variables[0].assign(tf.zeros_like(state[0])),
    #                             state_variables[1].assign(tf.zeros_like(state[1]))])
    #
    #     for state_variables, state in zip(cell_init_state_bw, states[1]):
    #         update_ops.extend([ state_variables[0].assign(tf.zeros_like(state[0])),
    #                             state_variables[1].assign(tf.zeros_like(state[1]))])
    #
    #     clear_state_op = tf.tuple(update_ops)
    #
    #     """ Define Policy and Value """
    #     with tf.variable_scope('core'):
    #         output_stack = tf.concat(output, 2)
    #         # atten_weights = tf.matmul(output_stack,output_stack,transpose_b=True)
    #         # attended_output = tf.reduce_sum(tf.expand_dims(atten_weights,axis=3)*tf.expand_dims(output_stack,axis=2),axis=2)
    #         # policy = tf.layers.dense(attended_output,self.k)
    #         policy = tf.layers.dense(output_stack,self.k)
    #         #policy = tf.layers.dense(tf.nn.relu(tf.layers.dense(output_stack,16)),self.k)
    #     """ Define Loss and Optimizer """

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels ,logits= policy))

        predicted_label = tf.argmax(policy,axis=2)
        miss_list_0 = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(predicted_label,tf.float64),tf.cast(labels,tf.float64)),tf.float32))

        miss_rate = miss_list_0/(self.num_sequence*self.batch_size)

        l2 = 0.001 * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if ("core" in tf_var.name)
        )
        policy_prob = tf.nn.softmax(policy,axis=2)
        policy_prob_stricter = tf.nn.softmax(tf.square(policy_prob),axis=2)

        cluster_centers = tf.reduce_mean(tf.expand_dims(policy_prob,axis=3)*tf.expand_dims(sequences,axis=2),axis=1,keepdims=True)
        diff_to_clusters = tf.norm(tf.expand_dims(sequences,axis=2) - cluster_centers,axis=3)
        diff_prob_to_clusters = tf.reduce_sum(tf.reduce_sum(diff_to_clusters*policy_prob,axis=1),axis=1)
        kmeans_loss = tf.reduce_mean(diff_prob_to_clusters)/self.num_sequence

        #opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize((self.alpha)*loss+(1-self.alpha)*kmeans_loss)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return AttrDict(locals())

    def mutual_info(self,true_label,predicted_label):
        nmi_list = []
        for i in range(true_label.shape[0]):
            nmi = normalized_mutual_info_score(true_label[i],predicted_label[i])
            nmi_list.append(nmi)
        return np.mean(nmi_list)

    def train(self,data,labels,sess):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(30):
            perm = np.random.permutation(self.num_sequence)
            data = data[:,perm,:]
            labels = labels[:,perm]
            _,_,miss_rate,predicted_label = sess.run([model.keep_state_op,model.opt,model.miss_rate,model.predicted_label],feed_dict={model.sequences:data,model.labels:labels})
            #miss_rate = sess.run([model.output],feed_dict={model.sequences:data,model.labels:labels})
            nmi = self.mutual_info(labels,predicted_label)
        print("Epochs{}: Miss rate {}, NMI {}".format(epoch_ind,miss_rate,nmi))

    def test(self,data,labels,sess,validation=False):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(30):
            perm = np.random.permutation(self.num_sequence)
            data = data[:,perm,:]
            labels = labels[:,perm]
            states,miss_rate,loss,predicted_label = sess.run([model.keep_state_op,model.miss_rate,model.loss,model.predicted_label],feed_dict={model.sequences:data,model.labels:labels})
            nmi = self.mutual_info(labels,predicted_label)
            if not validation:
                print("Epochs{}: Miss rate {}, NMI {}".format(epoch_ind,miss_rate,nmi))
        if validation:
            print("Epochs{}: Miss rate {}, NMI {}".format(epoch_ind,miss_rate,nmi))
        # if config.show_comparison_graph:
        #     data = np.squeeze(data)
        #     labels = np.squeeze(labels)
        #     predicted_label = np.squeeze(predicted_label)
        #     diff = np.abs(labels-predicted_label)
        #
        #     fig = plt.figure()
        #     ax = fig.add_subplot(311)
        #
        #     for i in range(self.k):
        #         ax.scatter(data[labels==i,0], data[labels==i,1])
        #     ax.set_title('Original',fontsize=8)
        #     #ax.axis('scaled')
        #
        #     ax = fig.add_subplot(312)
        #     for i in range(self.k):
        #         ax.scatter(data[predicted_label==i,0], data[predicted_label==i,1])
        #     ax.set_title('Predicton',fontsize=8)
        #     #ax.axis('scaled')
        #
        #     ax = fig.add_subplot(313)
        #     ax.scatter(data[diff==0,0], data[diff==0,1],color='black')
        #     ax.scatter(data[diff==1,0], data[diff==1,1],color='red')
        #     ax.set_title('Difference',fontsize=8)
        #     #ax.axis('scaled')
        #
        #     plt.show()

    def test_compare(self,data,labels,sess,kmeans, validation=False):
        model = self.model
        sess.run(model.clear_state_op)
        for epoch_ind in range(30):
            perm = np.random.permutation(self.num_sequence)
            data = data[:,perm,:]
            labels = labels[:,perm]
            states,miss_rate,loss,predicted_label = sess.run([model.keep_state_op,model.miss_rate,model.loss,model.predicted_label],feed_dict={model.sequences:data,model.labels:labels})
            nmi = self.mutual_info(labels,predicted_label)
            if not validation:
                print("Epochs{}: Miss rate {}, NMI {}".format(epoch_ind,miss_rate,nmi))
        if validation:
            print("Epochs{}: Miss rate {}, NMI {}".format(epoch_ind,miss_rate,nmi))

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
            ax.set_title('MetaCluster',fontsize=8)
            #ax.axis('scaled')

            ax = fig.add_subplot(313)
            for i in range(self.k):
                ax.scatter(data[kmeans.labels_==i,0], data[kmeans.labels_==i,1])
            ax.set_title('K-Means',fontsize=8)
            #ax.axis('scaled')

            plt.savefig('result.png')


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
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--fea', default=2, type=int)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--training_exp_num', default=50, type=int)

    config = parser.parse_args()

    if not config.test:
        metaCluster = MetaCluster(config)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # training
            for train_ind in tqdm(range(int(config.training_exp_num))):
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

            # generator = Generator_minst()
            # data, labels = generator.generate(metaCluster.num_sequence//2, metaCluster.fea)
            #
            # #data, labels = eduGenerate()
            #
            # #data, labels = make_circles(100)
            # #data, labels = make_moons(100)
            # kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            # print(metaCluster.mutual_info(np.expand_dims(labels,axis=0),np.expand_dims(kmeans.labels_,axis=0)))
            # print(np.sum(np.abs(labels-kmeans.labels_)))
            #
            # data = np.expand_dims(data, axis=0)
            # labels = np.expand_dims(labels, axis=0)
            #data, labels = metaCluster.create_dataset()

            from sklearn.datasets import load_iris


            iris = load_iris()
            data = iris.data#/np.max(iris.data)-0.5
            labels = iris.target
            x_norm = StandardScaler().fit_transform(data)
            pca = PCA(n_components=metaCluster.fea, whiten=True)
            data_pca = pca.fit_transform(x_norm)

            kmeans = KMeans(n_clusters=3, random_state=0).fit(data_pca)
            print(metaCluster.mutual_info(np.expand_dims(labels,axis=0),np.expand_dims(kmeans.labels_,axis=0)))
            metaCluster.test_compare(np.expand_dims(data_pca,axis=0),np.expand_dims(labels,axis=0),sess,kmeans)
