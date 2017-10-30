from glob import glob
import os
import time 
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset, Iterator
import numpy as np

import config
import models

train_dir = config.data_train_dir
val_dir = config.data_val_dir

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

    val_files = glob(os.path.join(val_dir, '*.tfrecord'))
    val_data = tf.contrib.data.TFRecordDataset(val_files)
    val_data = val_data.map(parser)
    val_data = val_data.batch(config.batch)
    val_iterator = Iterator.from_dataset(val_data)
    val_iterator_init_op = iterator.make_initializer(val_data)
    next_val_batch = iterator.get_next()

    saver = tf.train.Saver()
    if ():
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
    batch_num = 0
    best_val_loss = 1000.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)
        sess.run(val_iterator_init_op)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment) + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path and not config.restart:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path

        for i in range(config.num_epochs):
            while True:
                try:
                    batch = sess.run(next_batch)
                    signals, labels, sig_length, base_length = batch['signals'], batch['labels'], batch['sig_length'], batch['base_length'] 
                    signals = signals.astype(np.float32)
                    normalized = (signals - np.mean(signals, axis=1, keepdims=True))/np.std(signals, axis=1, keepdims=True)
                    _, loss_batch = sess.run([model.opt, model.loss], 
                        feed_dict={model.signals: normalized, model.labels:labels, model.sig_length:sig_length, model.base_length:base_length})
                    print 'Batch Loss is ', loss_batch 

                    if batch_num % config.val_every == 0: #Perform a validation every config.val_every batches
                        try:
                            batch = sess.run(next_val_batch)
                            signals, labels, sig_length, base_length = batch['signals'], batch['labels'], batch['sig_length'], batch['base_length'] 
                            signals = signals.astype(np.float32)
                            normalized = (signals - np.mean(signals, axis=1, keepdims=True))/np.std(signals, axis=1, keepdims=True)
                            val_loss_batch = sess.run([model.loss], 
                                feed_dict={model.signals: normalized, model.labels:labels, model.sig_length:sig_length, model.base_length:base_length})[0]
                            print val_loss_batch, best_val_loss
                            if val_loss_batch < best_val_loss:
                                save_path = saver.save(sess, os.path.join(config.save_dir, config.model, config.experiment, str(batch_num)) + '.ckpt')
                                print("Model saved in file: %s" % save_path)
                                best_val_loss = val_loss_batch
                            print 'Val Batch Loss is ', val_loss_batch
                            print 'Best Val Batch Loss is', best_val_loss
                        except tf.errors.OutOfRangeError:
                            print('No more batches in validation set, wait until next epoch')
                    batch_num += 1
                except tf.errors.OutOfRangeError:
                    print('End of Epoch ' + str(i+1))
                    batch_num = 0
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