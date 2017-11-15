import os
import tensorflow as tf
import config
import tensorflow.contrib.layers as layers
from tensorflow.python.layers import core as core_layers

#Parent class for all of the seq2seq models
class seq2seqModel:
    def __init__(self):
        self.logits = None
        self.predictions = None
        self.mask = None #Used to mask out padded positions

    def create_placeholder(self):
        with tf.name_scope('data'):
            self.signals = tf.placeholder(tf.float32, [None, None], name="signals_placeholder")
            self.labels = tf.placeholder(tf.int32, [None, None], name="label_placeholder")
            self.sig_length = tf.placeholder(tf.int32, [None], name='sig_length_placeholder')
            self.base_length = tf.placeholder(tf.int32, [None], name='base_length_placeholder')
            self.dropout_keep = tf.placeholder(tf.float32, [], name='dropout_keep')
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def inference(self):
        pass

    def _loss(self):
        pass

    def train_op(self):
        pass

    def _summary(self):
        with tf.name_scope("summaries"): #This adds the loss to tensorboard
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def setup_embeddings(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
            self.base_embeddings = tf.get_variable("base_embeddings", [6, 100]) #6 classes (4 bases, stop, start), 100 element vectors

    def build_train_graph(self):
        self.create_placeholder()
        self.setup_embeddings('embeddings')
        feed_labels = tf.concat((tf.expand_dims(tf.fill([config.batch], 5), 1), self.labels[:, :-1]), axis=1) 
        helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(self.base_embeddings, feed_labels),
                self.base_length, time_major=False)
        self.inference(helper)
        self._loss()
        self.train_op()
        self._summary()

    def build_val_graph(self):
        self.create_placeholder()
        self.setup_embeddings('embeddings')
        self.inference(tf.contrib.seq2seq.GreedyEmbeddingHelper(self.base_embeddings,
                start_tokens=tf.fill([config.batch], 5), end_token=4))
        self._loss()
        self.train_op()
        self._summary()

class BabyAchilles(seq2seqModel):
    #Same ResBlock as described in the Chiron Paper
    def resBlock(self, inputs, module_scope):
        with tf.variable_scope(module_scope) as scope:
            # if module_scope == 'res1': #We only need to make it 3 dimensional on the first block
                # inputs = tf.expand_dims(inputs, axis=-1)
            conv1 = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same', activation=tf.nn.relu)
            conv1a = tf.layers.conv1d(inputs, filters=256, kernel_size=1, strides=1, padding='same')
            conv2 = tf.layers.conv1d(conv1, filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', activation=tf.nn.relu)
            conv3 = tf.layers.conv1d(conv2, filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')
            return tf.nn.relu(conv3 + conv1a)
    #1 ResBlock and 1 BiRNN
    def encode(self, module_scope): 
        with tf.variable_scope(module_scope) as scope:
            self.conv_encoding = self.resBlock(tf.expand_dims(self.signals, axis=-1), 'res1')
            forward = tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size)
            backward = tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size)
            bi_outputs, last_encoder_state = tf.nn.bidirectional_dynamic_rnn(forward, backward, self.conv_encoding, sequence_length=self.sig_length, time_major=False, dtype=tf.float32)
            bi_outputs = tf.concat(bi_outputs, -1) #Fuse both directions
            c1, h1, c2, h2 = last_encoder_state[0].c, last_encoder_state[0].h, last_encoder_state[1].c, last_encoder_state[1].h
            last_encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.concat((c1, c2), -1), tf.concat((h1, h2), -1))
            return (bi_outputs, last_encoder_state)
    #Attention Decoder
    def decode(self, encoding_tup, helper, module_scope):
        with tf.variable_scope(module_scope) as scope:
            encoding, last_encoder_state = encoding_tup
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=200)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=200, memory=encoding, memory_sequence_length=self.sig_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=50)
            
            initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=config.batch)
            initial_state = initial_state.clone(cell_state=last_encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer=core_layers.Dense(6, use_bias=False))
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=config.max_base_len) #Really confused about this!
            self.logits = outputs.rnn_output
            self.predictions = outputs.sample_id
    #Masked CE loss
    def _loss(self):
        with tf.variable_scope('loss') as scope:
            self.mask = tf.to_float(tf.not_equal(self.labels, 4)) #Mask out the padded outputs
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_sum(crossent * self.mask) / config.batch
    #Adam optimizer and Clipped Gradients
    def train_op(self):
        with tf.variable_scope('train') as scope:
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.opt = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def inference(self, helper):
        encoding_tup = self.encode('encode')
        self.decode(encoding_tup, helper, 'decode')

class Achilles(BabyAchilles):
    #3 ResBlock -> 3 stacked BiRNN
    def encode(self, module_scope): 
        with tf.variable_scope(module_scope) as scope:
            res1 = self.resBlock(tf.expand_dims(self.signals, axis=-1), 'res1') #Lets get a representation of each time step first
            res1_batch = tf.contrib.layers.batch_norm(res1, center=True, scale=True, is_training=True, scope='bn1')
            res2 = self.resBlock(res1_batch, 'res2')
            res2_batch = tf.contrib.layers.batch_norm(res2, center=True, scale=True, is_training=True, scope='bn2')
            res3 = self.resBlock(res2_batch, 'res3')
            res3_batch = tf.contrib.layers.batch_norm(res3, center=True, scale=True, is_training=True, scope='bn3')
            res3 = tf.nn.dropout(res3, self.dropout_keep) #dropout on the last layer to give noisier inputs to rnn
                                                       
            forward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #Then let's get a representation of the entire series!
            backward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #With a 3 layer lstm

            outputs, last_for, last_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cells, backward_cells, res3, 
                sequence_length=self.sig_length, dtype=tf.float32)
            c_for, h_for, c_back, h_back = last_for[2].c, last_for[2].h, last_back[2].c, last_back[2].h #Grab the last state from the last layer!
            last_encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.concat((c_for, c_back), -1), tf.concat((h_for, h_back), -1))
            return (outputs, last_encoder_state)