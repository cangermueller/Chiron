from glob import glob
import os
import time 
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset, Iterator
import numpy as np

import config
import models

train_dir = config.data_train_dir

def parser(record):
    context_features = {
        "sig_length": tf.FixedLenFeature([], dtype=tf.int64),
        "base_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return {'sig_length': context_parsed['sig_length'], 'base_length': context_parsed['base_length'],
        'signals':sequence_parsed['tokens'], 'labels':sequence_parsed['labels']}

def train(model):
    train_files = glob(os.path.join(train_dir, '*.tfrecord'))
    train_data = tf.contrib.data.TFRecordDataset(train_files)
    train_data = train_data.map(parser)
    train_data = train_data.batch(config.batch)

    iterator = Iterator.from_dataset(train_data)
    iterator_init_op = iterator.make_initializer(train_data)
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)
        for i in range(config.num_epochs):
            while True:
                try:
                    batch = sess.run(next_batch)
                    signals, labels, sig_length, base_length = batch['signals'], batch['labels'], batch['sig_length'], batch['base_length'] 
                    signals = signals.astype(np.float32)
                    normalized = (signals - np.mean(signals, axis=1, keepdims=True))/np.std(signals, axis=1, keepdims=True)
                    _, loss_batch = sess.run([model.opt, model.loss], 
                        feed_dict={model.signals: normalized, model.labels:labels, model.sig_length:sig_length, model.base_length:base_length})
                    print('Batch Loss is ', loss_batch) 
                except tf.errors.OutOfRangeError:
                    print('End of Epoch ' + str(i+1))
                    break

def main():
    model = None
    if config.model == 'Baseline':
        model = models.Baseline()
    if config.train:
        model.build_graph()
        train(model)

if __name__ == '__main__':
    main()