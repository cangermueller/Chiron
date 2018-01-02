import tensorflow as tf
import numpy as np
import config


###############################################################################
### layers, ops, etc
###############################################################################
def causal_transform(tensor, dilation=1, name=None):
    '''Reshape 1d time-series by dialtion, zero-pad if needed'''
    shape = tf.shape(tensor)
    out_width = tf.to_int32(shape[0] * dilation)
    _, _, out_channels = tensor.get_shape().as_list()
    padding = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(tensor, [[0, 0], [0, padding], [0, 0]])
    reshaped = tf.reshape(padded, (-1, dilation, out_channels))
    transposed = tf.transpose(reshaped, perm=(1, 0, 2))
    transformed = tf.reshape(transposed, (out_width, -1, out_channels))
    return transformed

def causal_restore(tensor, dilation=1, name=None):
    '''Reshape batch back to 1d time-series after dilation'''
    shape = tf.shape(tensor)
    out_width = tf.div(shape[0], dilation)
    _, _, out_channels = tensor.get_shape().as_list()
    reshaped = tf.reshape(tensor, (dilation, -1, out_channels))
    transposed = tf.transpose(reshaped, perm=(1, 0, 2))
    return tf.reshape(transposed, (out_width, -1, out_channels))
    
def causal_conv(tensor, filters=256, kernel_size=2, dilation=1,
                activation=None, use_bias=False, name='causal_conv'):
    with tf.variable_scope(name):
        input_shape = tensor.get_shape().as_list()
        # preserve causality by padding beforehand
        causal_padding = int(kernel_size - 1) * dilation
        padded = tf.pad(tensor, [[0, 0], [causal_padding, 0], [0, 0]])
        if dilation > 1:
            dilated = causal_transform(padded, dilation=dilation)
            dilated_conv = conv_layer(dilated,
                                            filters=filters,
                                            kernel_size=kernel_size, 
                                            strides=1, padding='SAME', 
                                            activation=activation,
                                            use_bias=use_bias)
            restored = causal_restore(dilated_conv, dilation=dilation) 
        else:
            # no dilations
            restored = conv_layer(padded,
                                        filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=1, padding='SAME', 
                                        activation=activation,
                                        use_bias=use_bias)
                                 
        # Add additional shape information.
        restored = tf.slice(restored,
                            [0, 0, 0],
                            [-1, tf.shape(tensor)[1], -1])
        output_shape = [tf.Dimension(None),
                        tf.Dimension(input_shape[1]),
                        tf.Dimension(filters)]
        restored.set_shape(tf.TensorShape(output_shape))
        return restored



def conv_layer(tensor, filters=256, kernel_size=2,
           strides=1, padding='SAME', gain=np.sqrt(2), 
           activation=None, use_bias=False, name='conv_layer'):
    '''One dimension convolution helper function.
       Sets variables with good defaults.
    '''   
    with tf.variable_scope(name) as scope:
        in_channels = tensor.get_shape().as_list()[-1]
        w = tf.get_variable(name='w',
                            shape=(kernel_size, in_channels, filters),
                            initializer=tf.random_normal_initializer(
                                stddev=gain / np.sqrt(kernel_size**2 * in_channels)
                            ))
        result = tf.nn.conv1d(tensor, w, stride=strides, padding=padding)
    if use_bias:
        b = tf.get_variable(name='b',
                            shape=(filters, ),
                            initializer=tf.constant_initializer(0.0))
        result = result + tf.expand_dims(tf.expand_dims(b, 0), 0)
    if activation:
        result = activation(result)
    return result

    
###############################################################################
### base models
###############################################################################
class Poseidon(object):
    '''Implements the WaveNet network for nanopore sequence basecalling.
    '''
    def __init__(self,
                 blocks=4,
                 layers=4,
                 classes=5,
                 filters=256,
                 kernel_size=2,
                 residual_filters=16,
                 skip_filters=16,
                 dilation_filters=32,
                 quantization_filters=16,
                 dilation_factor=2, 
                 activation=tf.nn.relu,
                 use_bn=False,
                 use_bias=False,
                 lstm_cells=3,
                 lstm_size=200,
                 max_seq_len=None,
                 batch_size=None,
                 config=config,
                 verbose=True):
   
        self.logits = None
        self.predictions = None
        self.edit_distances = None


        self.blocks = blocks   # ?
        self.layers = layers   # ?
        self.classes = classes # quantization channels

        self.filters = filters
        self.kernel_size = kernel_size                # kernel size?
        self.residual_filters = residual_filters 
        self.skip_filters = skip_filters 
        self.dilation_filters = dilation_filters
        self.quantization_filters = quantization_filters
        self.dilation_factor = dilation_factor
        
        self.activation = activation
        self.use_bn = use_bn
        self.use_bias = use_bias

        self.lstm_size = lstm_size
        self.lstm_cells = lstm_cells
        self.max_seq_len = max_seq_len or config.max_seq_len
        self.batch_size = batch_size or config.batch
        self.config = config 
        self.verbose = verbose
    
                               
    def create_placeholder(self):
        with tf.name_scope('data'):
            self.signals = tf.placeholder(tf.float32, [None, self.max_seq_len], name="signals")
            
            # CTC expects labels as SparseTensor
            self.y_indices = tf.placeholder(tf.int64, name="y_indices")
            self.y_values = tf.placeholder(tf.int32, name="y_values")
            self.y_shape = tf.placeholder(tf.int64, name="y_shape") 
            
            self.sig_length = tf.placeholder(tf.int32, [None], name='sig_length')
            self.base_length = tf.placeholder(tf.int32, [None], name='base_length')
            self.dropout_keep = tf.placeholder(tf.float32, [], name='dropout_keep')
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')


            
    def _loss(self):
        with tf.variable_scope('loss') as scope:
            labels = tf.SparseTensor(self.y_indices,self.y_values,self.y_shape)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels, self.logits, self.sig_length, 
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=False,
                time_major=False))

    def train_op(self):
        with tf.variable_scope('train') as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
                self.opt = tf.train.AdamOptimizer(self.lr).apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step)

    def _summary(self):
        with tf.name_scope("summaries"): #This adds the loss to tensorboard
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_summary(self):
        with tf.name_scope("summaries"): #This adds the loss to tensorboard
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def val_summary(self):
        with tf.name_scope("summaries"): #This adds the loss to tensorboard
            tf.summary.scalar("loss", self.loss)
            #tf.summary.scalar("error", self.error)
            #tf.summary.scalar("mean_edit_distance", self.mean_distances)
            # add some tensor summaries for fun, not really sure what they are tho...
            #tf.contrib.layers.summarize_tensor(self.signals, tag="signals")
            tf.contrib.layers.summarize_tensor(self.logits, tag="logits")
            tf.contrib.layers.summarize_tensor(self.edit_distances, tag="edit_distances")
            self.summary_op = tf.summary.merge_all()

           

    def build_train_graph(self):
        self.create_placeholder()
        self.inference()
        self._loss()
        self.train_op()
        self.train_summary()

    def build_val_graph(self):
        self.create_placeholder()
        self.inference()
        self._loss()
        self.val_summary()

     
            
    def inference(self):
        self.encode('encode')
        self.predict('predict')


    def predict(self, module_scope):
        with tf.variable_scope(module_scope):
            # prediction
            self.predictions = tf.nn.ctc_beam_search_decoder(
                tf.transpose(self.logits, perm=[1,0,2]), self.sig_length,
                merge_repeated=False, beam_width=3)
            # compute error
            labels = tf.SparseTensor(self.y_indices,self.y_values,self.y_shape)
            self.predicted_bases = tf.to_int32(self.predictions[0][0])
            self.edit_distances = tf.edit_distance(self.predicted_bases, labels, normalize=False)
            self.error = tf.reduce_sum(self.edit_distances) / tf.to_float(tf.size(labels.values))
 
               
    def wavenet_layer(self, layer_input, dilation=1, name='wavenet_layer'):
        """Creates a single Wavenet causal dilated convolution layer."""
        with tf.variable_scope(name):
            conv_filter = causal_conv(layer_input, self.dilation_filters,
                                      self.kernel_size, dilation=dilation,
                                      name='conv_filter')
            conv_gate = causal_conv(layer_input, self.dilation_filters,
                                    self.kernel_size, dilation=dilation, 
                                    name='conv_gate')
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
            residual = conv_layer(out,
                                  self.residual_filters,
                                  kernel_size=1,
                                  activation=self.activation,
                                  use_bias=self.use_bias,
                                  name='residual')
            skip_connection = conv_layer(out,
                                          self.skip_filters,
                                          kernel_size=1,
                                          activation=self.activation,
                                          use_bias=self.use_bias,
                                          name='skip_connection')
            return layer_input + residual, skip_connection

            
    def wavenet_block(self, current_layer, name='wavenet_block'):
        """Creates a block of Wavenet causal dilated convolution layers."""
        block_output = 0
        for l in range(self.layers):
            current_layer, skip = self.wavenet_layer(
                current_layer,
                dilation=self.dilation_factor**l,
                name=name+'/layer-%s'%l)
            block_output += skip 
        return current_layer, block_output
      
    def wavenet_stack(self, input_batch, name='wavenet_stack'):
        '''Creates the Wavenet network, and applies to input_batch.
        '''
        with tf.name_scope(name):
            stack_output = 0
            current_block = input_batch

            # 1. preserve causality in signal
            current_block = causal_conv(current_block,
                                        self.residual_filters,
                                        kernel_size=self.kernel_size,
                                        dilation=1)
            
            # 2. collect skip output
            for b in range(self.blocks):
                current_block, skip = self.wavenet_block(current_block, name='block-%s'%b)
                stack_output += skip 

            # 3. process: (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv
            active_1 = self.activation(stack_output)
            conv_1 = conv_layer(active_1,
                                      filters=self.skip_filters,
                                      kernel_size=1, activation=None,
                                      strides=1, padding='SAME',
                                      use_bias=self.use_bias,
                                      name='conv_1')
            active_2 = self.activation(conv_1)
            conv_2 = conv_layer(active_2,
                            filters=self.quantization_filters,
                            kernel_size=1, activation=None,
                            strides=1, padding='SAME',
                            use_bias=self.use_bias,
                            name='conv_2')
            return conv_2

    
    def lstm_stack(self, input_batch, name='lstm_stack'):
        '''Create and apply bi-directional LSTM to input batch
        '''
        #  representation of entire series! (with 3-layer lstm)
        with tf.name_scope(name):
            forward_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size) for i in range(self.lstm_cells)] 
            backward_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size) for i in range(self.lstm_cells)] 
            output, last_for, last_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
              forward_cells, backward_cells, input_batch, sequence_length=self.sig_length, dtype=tf.float32)
            output_rs = tf.reshape(output, (-1, 2*self.lstm_size))
            return output_rs

    
    def encode(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
            # reshape layer
            raw_signal = tf.expand_dims(self.signals, axis=-1)

            # 1. CNN stack, dropout for noisier rnn inputs
            cnn_output = self.wavenet_stack(raw_signal)
            cnn_dropout = tf.nn.dropout(cnn_output, self.dropout_keep)

            # 2. RNN stack
            rnn_output = self.lstm_stack(cnn_dropout)    

            # final dense, reshape layers
            dense_output = tf.layers.dense(rnn_output, self.classes)
            self.logits = tf.reshape(dense_output, (-1, self.max_seq_len, self.classes))


###############################################################################
### subclassed models
###############################################################################
class Triton(Poseidon):
    '''Wavenet Model with Chiron's cnn stack.
    '''    
    def __init__(self, **kwargs):
        Poseidon.__init__(self, **kwargs)

    def chiron_block(self, input_batch, name='chiron_block'):
        with tf.variable_scope(name):
            conv1 = tf.layers.conv1d(input_batch, filters=self.filters,
                                     kernel_size=1, strides=1, use_bias=False,
                                     padding='same', activation=tf.nn.relu)
            conv1a = tf.layers.conv1d(input_batch, filters=self.filters,
                                      kernel_size=1, strides=1, padding='same')
            conv2 = tf.layers.conv1d(conv1, filters=self.filters,
                                     kernel_size=3, strides=1, use_bias=False,
                                     padding='same', activation=tf.nn.relu)
            conv3 = tf.layers.conv1d(conv2, filters=self.filters,
                                     kernel_size=1, strides=1, use_bias=False,
                                     padding='same')
            return tf.nn.relu(conv3 + conv1a)
        
    def chiron_stack(self, input_batch, name='chiron_stack'):
        with tf.name_scope(name):
            current_layer = input_batch
            for b in range(self.blocks):
                current_layer = self.chiron_block(current_layer, name='res-%d'%b)
            return current_layer

    
    def encode(self, module_scope):
        with tf.variable_scope(module_scope) as scope:
            # reshape layer
            raw_signal = tf.expand_dims(self.signals, axis=-1)

            # 1. CNN stacks (Wavenet->Chiron), dropout for noisier rnn inputs
            cnn_output_1 = self.wavenet_stack(raw_signal)
            cnn_output_2 = self.chiron_stack(cnn_output_1)
            cnn_dropout = tf.nn.dropout(cnn_output_2, self.dropout_keep)

            # 2. RNN stack
            rnn_output = self.lstm_stack(cnn_dropout)    

            # final dense, reshape layers
            dense_output = tf.layers.dense(rnn_output, self.classes)
            self.logits = tf.reshape(dense_output, (-1, self.max_seq_len, self.classes))

