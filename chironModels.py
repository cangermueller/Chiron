import os
import tensorflow as tf
import config
# import tensorflow.contrib.layers as layers
from tensorflow.python.layers import core as core_layers

class baseChironModel():
    def __init__(self):
        self.logits = None
        self.predictions = None

    def create_placeholder(self):
        with tf.name_scope('data'):
            self.signals = tf.placeholder(tf.float32, [None, config.max_seq_len], name="signals_placeholder")

            self.y_indices = tf.placeholder(tf.int64)
            self.y_values = tf.placeholder(tf.int32)
            self.y_shape = tf.placeholder(tf.int64)  

            self.sig_length = tf.placeholder(tf.int32, [None], name='sig_length_placeholder')
            self.base_length = tf.placeholder(tf.int32, [None], name='base_length_placeholder')
            self.dropout_keep = tf.placeholder(tf.float32, [], name='dropout_keep')
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def resBlock(self, inputs, module_scope):
        with tf.variable_scope(module_scope) as scope:
            conv1 = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same', activation=tf.nn.relu)
            conv1a = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, padding='same')
            conv2 = tf.layers.conv1d(conv1, filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', activation=tf.nn.relu)
            conv3 = tf.layers.conv1d(conv2, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')
            return tf.nn.relu(conv3 + conv1a)

    def inference(self):
        pass

    def _loss(self):
        with tf.variable_scope('loss') as scope:
            labels = tf.SparseTensor(self.y_indices, self.y_values, self.y_shape)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels, self.logits, self.sig_length, preprocess_collapse_repeated=False, ctc_merge_repeated=False, time_major=False))

    def train_op(self):
        with tf.variable_scope('train') as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
                self.opt = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def _summary(self):
        with tf.name_scope("summaries"): #This adds the loss to tensorboard
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_train_graph(self):
        self.create_placeholder()
        self.inference()
        self._loss()
        self.train_op()
        self._summary()

    def build_val_graph(self):
        self.create_placeholder()
        self.inference()
        self._loss()
        self._summary()

class Chiron_3(baseChironModel):
    def encode(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
                res1 = self.resBlock(tf.expand_dims(self.signals, axis=-1), 'res1') #Lets get a representation of each time step first
                res1_batch = tf.layers.batch_normalization(res1, axis=-1, center=True, scale=True, training=self.is_training)
                res2 = self.resBlock(res1_batch, 'res2')
                res2_batch = tf.layers.batch_normalization(res2, axis=-1, center=True, scale=True, training=self.is_training)            
                res3 = self.resBlock(res2_batch, 'res3')
                res3_batch = tf.layers.batch_normalization(res3, axis=-1, center=True, scale=True, training=self.is_training)
                res3_drop = tf.nn.dropout(res3_batch, self.dropout_keep) #dropout on the last layer to give noisier inputs to rnn                                    
                forward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #Then let's get a representation of the entire series!
                backward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #With a 3 layer lstm
                outputs, last_for, last_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cells, backward_cells, res3_drop, 
                    sequence_length=self.sig_length, dtype=tf.float32)
                outputs_r = tf.reshape(outputs, (-1, 2*config.lstm_size)) #B*T, S
                self.logits = tf.layers.dense(outputs_r, 5)
                self.logits = tf.reshape(self.logits, (-1, config.max_seq_len, 5))
    def inference(self):
        self.encode('encode')
        # self.predictions = tf.nn.ctc_greedy_decoder(tf.transpose(self.logits, perm=[1,0,2]), self.sig_length, merge_repeated = False)
        self.predictions = tf.nn.ctc_beam_search_decoder(tf.transpose(self.logits, perm=[1,0,2]), self.sig_length, merge_repeated = False, beam_width=3)

class Chiron_5(baseChironModel):
    def encode(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
                res1 = self.resBlock(tf.expand_dims(self.signals, axis=-1), 'res1') #Lets get a representation of each time step first
                res1_batch = tf.layers.batch_normalization(res1, axis=-1, center=True, scale=True, training=self.is_training)
                res2 = self.resBlock(res1_batch, 'res2')
                res2_batch = tf.layers.batch_normalization(res2, axis=-1, center=True, scale=True, training=self.is_training)            
                res3 = self.resBlock(res2_batch, 'res3')
                res3_batch = tf.layers.batch_normalization(res3, axis=-1, center=True, scale=True, training=self.is_training)
                res4 = self.resBlock(res3_batch, 'res4')
                res4_batch = tf.layers.batch_normalization(res4, axis=-1, center=True, scale=True, training=self.is_training)
                res5 = self.resBlock(res4_batch, 'res5')
                res5_batch = tf.layers.batch_normalization(res5, axis=-1, center=True, scale=True, training=self.is_training)
                res5_drop = tf.nn.dropout(res5_batch, self.dropout_keep) #dropout on the last layer to give noisier inputs to rnn                                    
                forward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #Then let's get a representation of the entire series!
                backward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #With a 3 layer lstm
                outputs, last_for, last_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cells, backward_cells, res5_drop, 
                    sequence_length=self.sig_length, dtype=tf.float32)
                outputs_r = tf.reshape(outputs, (-1, 2*config.lstm_size)) #B*T, S
                self.logits = tf.layers.dense(outputs_r, 5)
                self.logits = tf.reshape(self.logits, (-1, config.max_seq_len, 5))
    def inference(self):
        self.encode('encode')
        # self.predictions = tf.nn.ctc_greedy_decoder(tf.transpose(self.logits, perm=[1,0,2]), self.sig_length, merge_repeated = False)
        self.predictions = tf.nn.ctc_beam_search_decoder(tf.transpose(self.logits, perm=[1,0,2]), self.sig_length, merge_repeated = False, beam_width=3)