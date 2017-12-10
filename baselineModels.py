import os
import tensorflow as tf
import config
# import tensorflow.contrib.layers as layers
from tensorflow.python.layers import core as core_layers

class baseBaselineModel():
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

    def inference(self):
        pass

    def _loss(self):
        with tf.variable_scope('loss') as scope:
            labels = tf.SparseTensor(self.y_indices, self.y_values, self.y_shape)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels, self.logits, self.sig_length, 
                preprocess_collapse_repeated=False, ctc_merge_repeated=False, time_major=False))

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

class Baseline(baseBaselineModel):
    def encode(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
                signal_batch = tf.layers.batch_normalization(tf.expand_dims(self.signals, axis=-1), axis=-1, center=True, scale=True, training=self.is_training)
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) # Single "forward" lstm layer
                outputs, state = tf.nn.dynamic_rnn(lstm_cell, signal_batch, sequence_length=self.sig_length, dtype=tf.float32)
                outputs_r = tf.reshape(outputs, (-1, config.lstm_size)) #B*T, S
                self.logits = tf.layers.dense(outputs_r, 5)
                self.logits = tf.reshape(self.logits, (-1, config.max_seq_len, 5))
    def inference(self):
        self.encode('encode')
        # self.predictions = tf.nn.ctc_greedy_decoder(tf.transpose(self.logits, perm=[1,0,2]), self.sig_length, merge_repeated = False)
        self.predictions = tf.nn.ctc_beam_search_decoder(tf.transpose(self.logits, perm=[1,0,2]),
            self.sig_length, merge_repeated = False, beam_width=3)

