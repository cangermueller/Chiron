import os
import tensorflow as tf
import config
import tensorflow.contrib.layers as layers
from tensorflow.python.layers import core as core_layers

import models

class SuperBaseline(models.Baseline):
    def encode(self, module_scope): 
        with tf.variable_scope(module_scope) as scope:
            res1 = self.resBlock(self.signals, 'res1') #Lets get a representation of each time step first
            res2 = self.resBlock(res1, 'res2')
            res3 = self.resBlock(res2, 'res3')

            res3 = tf.nn.dropout(res3, self.dropout_keep) #dropout on the last layer to give noisier inputs to rnn
            										   
            forward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #Then let's get a representation of the entire series!
            backward_cells = [tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size) for i in range(3)] #With a 3 layer lstm

            outputs, last_for, last_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cells, backward_cells, res3, 
            	sequence_length=self.sig_length, dtype=tf.float32)
            c_for, h_for, c_back, h_back = last_for[2].c, last_for[2].h, last_back[2].c, last_back[2].h #Grab the last state from the last layer!
            last_encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.concat((c_for, c_back), -1), tf.concat((h_for, h_back), -1))
            return (outputs, last_encoder_state)