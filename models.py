import os
import tensorflow as tf
import config
import tensorflow.contrib.layers as layers

class Model:
    def __init__(self, training=True):
        self.training = training

    def _create_placeholder(self):
        with tf.name_scope('data'):
            self.signals = tf.placeholder(tf.float32, [None, config.max_seq_len], name="signals_placeholder")
            self.labels = tf.placeholder(tf.int32, [None, config.max_base_len], name="label_placeholder")
            self.sig_length = tf.placeholder(tf.int32, [None], name='sig_length_placeholder')
            self.base_length = tf.placeholder(tf.int32, [None], name='base_length_placeholder')
    
    def _inference(self):
        pass

    def _loss(self):
        pass

    def _train_op(self):
        pass

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._loss()
        self._train_op()

class Baseline(Model):
    def resBlock(self, inputs, module_scope = ''):
        with tf.variable_scope(module_scope) as scope:
            inputs = tf.expand_dims(inputs, axis=-1)
            conv1 = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')
            conv1a = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, padding='same')
            conv2 = tf.layers.conv1d(conv1, filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')
            conv3 = tf.layers.conv1d(conv2, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')
            return tf.nn.relu(conv3 + conv1a)

    def _encode(self, module_scope): #Just 1 residual block for now
        with tf.variable_scope(module_scope):
            conv_encoding = self.resBlock(self.signals, 'res1')
            forward = tf.nn.rnn_cell.BasicLSTMCell(100)
            backward = tf.nn.rnn_cell.BasicLSTMCell(100)
            bi_outputs, last_encoder_state = tf.nn.bidirectional_dynamic_rnn(forward, backward, conv_encoding, sequence_length=self.sig_length, time_major=False, dtype=tf.float32)
            bi_outputs = tf.concat(bi_outputs, -1) #Fuse both directions
            c1, h1, c2, h2 = last_encoder_state[0].c, last_encoder_state[0].h, last_encoder_state[1].c, last_encoder_state[1].h
            last_encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.concat((c1, c2), -1), tf.concat((h1, h2), -1))
            return (bi_outputs, last_encoder_state)

    def _decode(self, encoding_tup, module_scope):
        with tf.variable_scope(module_scope):
            encoding, last_encoder_state = encoding_tup
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=200)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=200, memory=encoding, memory_sequence_length=self.sig_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=50)
            
            initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=config.batch)
            initial_state = initial_state.clone(cell_state=last_encoder_state)
            
            helper = tf.contrib.seq2seq.TrainingHelper(encoding, self.sig_length, time_major=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=config.max_base_len)

            logits = tf.layers.dense(tf.reshape(outputs.rnn_output, (-1, 50)), 5) #(B*T, 5)
            self.logits = tf.reshape(logits, (-1, config.max_base_len, 5)) #(B, 75, 5)

    def _loss(self):
        mask = tf.sequence_mask(self.base_length, maxlen=config.max_base_len, dtype=tf.float32)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_sum(mask*crossent) / config.batch

    def _train_op(self):
        self.opt = tf.train.AdamOptimizer(config.lr).minimize(self.loss)

    def _inference(self):
        encoding_tup = self._encode('encode')
        self._decode(encoding_tup, 'decode')
        self._loss()