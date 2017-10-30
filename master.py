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

def getDatasetIterator(sig, lab):
    sig_files = glob(sig)
    lab_files = glob(lab)
    sig_dataset = tf.contrib.data.TextLineDataset(sig_files)
    sig_dataset = sig_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    sig_dataset = sig_dataset.map(lambda sequence: (sequence, tf.size(sequence)))
    sig_dataset = sig_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.float32), length))
    lab_dataset = tf.contrib.data.TextLineDataset(lab_files)
    lab_dataset = lab_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    lab_dataset = lab_dataset.map(lambda sequence: (sequence, tf.size(sequence)))
    lab_dataset = lab_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.int32), length))
    dataset = tf.contrib.data.Dataset.zip((sig_dataset, lab_dataset))
    print dataset
    batched_dataset = dataset.padded_batch(
        config.batch,
        padded_shapes=((tf.TensorShape([config.max_seq_len]),  # source vectors of unknown size
                        tf.TensorShape([])),     # size(source)
                       (tf.TensorShape([config.max_base_len]),  # target vectors of unknown size
                        tf.TensorShape([]))),    # size(target)
        padding_values=((0.0, 0), (4, 0))) #Pad signal with 0.0 and label with 4  and don't pad the two lengths
    batched_iterator = batched_dataset.make_initializable_iterator()   
    return batched_iterator

def train(model):
    train_iterator = getDatasetIterator(os.path.join(train_dir, '*signal.txt'), os.path.join(train_dir, '*label.txt'))
    val_iterator = getDatasetIterator(os.path.join(val_dir, '*signal.txt'), os.path.join(val_dir, '*label.txt'))
    train_batch = train_iterator.get_next()
    val_batch = val_iterator.get_next()

    saver = tf.train.Saver()
    batch_num = 0
    best_val_loss = 1000.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment) + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path and not config.restart:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path

        for i in range(config.num_epochs):
            while True:
                try:
                    batch = sess.run(train_batch)
                    signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 
                    _, loss_batch = sess.run([model.opt, model.loss], 
                        feed_dict={model.signals: signals, model.labels:labels, model.sig_length:sig_length, model.base_length:base_length})
                    print 'Batch Loss is ', loss_batch 

                    if batch_num % config.val_every == 0: #Perform a validation every config.val_every batches
                        try:
                            batch = sess.run(val_batch)
                            signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 
                            val_loss_batch = sess.run([model.loss], 
                                feed_dict={model.signals: signals, model.labels:labels, model.sig_length:sig_length, model.base_length:base_length})[0]
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