from glob import glob
import os
import time 
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset, Iterator
import numpy as np

import config
import models
import super_baseline
import history

train_dir = config.data_train_dir
val_dir = config.data_val_dir

def getDatasetIterator(sig):
    sig_files = glob(sig)
    lab_files = [f[:-11] + '_label.txt' for f in sig_files]
    sig_dataset = tf.contrib.data.TextLineDataset(sig_files)
    sig_dataset = sig_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    sig_dataset = sig_dataset.map(lambda sequence: (sequence, tf.size(sequence))) #Dont forget about this!!!
    sig_dataset = sig_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.float32), length))
    lab_dataset = tf.contrib.data.TextLineDataset(lab_files)
    lab_dataset = lab_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    lab_dataset = lab_dataset.map(lambda sequence: (sequence, tf.size(sequence)))
    lab_dataset = lab_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.int32), length))
    dataset = tf.contrib.data.Dataset.zip((sig_dataset, lab_dataset))

    batched_dataset = dataset.padded_batch(
        config.batch,
        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                        tf.TensorShape([])),     # size(source)
                       (tf.TensorShape([None]),  # target vectors of unknown size
                        tf.TensorShape([]))),    # size(target)
        padding_values=((0.0, 0), (4, 0)))       #Pad signal with 0.0 and label with 4 and don't pad the two lengths
    batched_iterator = batched_dataset.make_initializable_iterator()   
    return batched_iterator

def train(model):
    train_iterator = getDatasetIterator(os.path.join(train_dir, '*signal.txt'))
    val_iterator = getDatasetIterator(os.path.join(val_dir, '*signal.txt'))
    train_batch = train_iterator.get_next()
    val_batch = val_iterator.get_next()

    saver = tf.train.Saver()
    batch_num = 0
    best_val_loss = 1000.0

    train_his = history.History(config.lr, config.dropout_keep)
    val_his = history.History(config.lr, config.dropout_keep)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) #Apparently needed for accuracy, and recall etc. to be initialized?

        writer = tf.summary.FileWriter(os.path.join(config.save_dir, config.model, config.experiment), sess.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment) + '/checkpoint'))

        if ckpt and ckpt.model_checkpoint_path and not config.restart:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path

        for i in range(1, config.num_epochs+1):
            sess.run(train_iterator.initializer)
            sess.run(val_iterator.initializer)
            while True:
                try:
                    batch = sess.run(train_batch)
                    signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 

                    if len(signals) != config.batch: #We really need exactly batch number of samples
                        continue
                    
                    _, loss_batch, train_acc, train_recall, train_prec = sess.run([model.opt, model.loss, model.accuracy, model.recall, model.precision], 
                        feed_dict={model.signals: signals, model.labels: labels, model.sig_length: sig_length, 
                            model.base_length: base_length, model.dropout_keep: config.dropout_keep, model.is_training:True})
                    print 'Batch Loss is ', loss_batch 
                    train_his.update(loss_batch, train_acc, train_recall, train_prec)
                    if batch_num % config.val_every == 0: #Perform a validation every config.val_every batches
                        try:
                            batch = sess.run(val_batch)
                            signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 
                            val_loss_batch, val_acc, val_prec, val_recall = sess.run([model.loss, model.accuracy, model.precision, model.recall], 
                                feed_dict={model.signals: signals, model.labels: labels, model.sig_length: sig_length, 
                                model.base_length: base_length, model.dropout_keep: 1.0, model.is_training:False})
                            if val_loss_batch < best_val_loss:
                                save_path = saver.save(sess, os.path.join(config.save_dir, config.model, config.experiment, str(batch_num)) + '.ckpt')
                                print("Model saved in file: %s" % save_path)
                                best_val_loss = val_loss_batch
                            print 'Val Batch Loss is ', val_loss_batch
                            print 'Best Val Batch Loss is', best_val_loss
                            val_his.update(val_loss_batch, val_acc, val_prec, val_recall)
                        except tf.errors.OutOfRangeError:
                            print 'No more batches in validation set, wait until next epoch'
                    batch_num += 1
                except tf.errors.OutOfRangeError:
                    print 'End of Epoch ' + str(i)
                    batch_num = 0
                    break
        writer.close()
        train_his.dump(os.path.join(config.save_dir, config.model, config.experiment, 'train.csv'))
        val_his.dump(os.path.join(config.save_dir, config.model, config.experiment, 'val.csv'))

def pred(model):
    pred_iterator = getDatasetIterator(os.path.join(config.pred_dir, '*signal.txt'))
    pred_batch = pred_iterator.get_next()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(pred_iterator.initializer)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment) + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path

        while True:
            try:
                batch = sess.run(pred_batch) #This works for val, but we will need to rework for test because we have no labels!
                signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 

                if len(signals) != config.batch: #We really need exactly batch number of samples
                        continue

                batch_predictions = sess.run([model.predictions], feed_dict={model.signals: signals, model.labels: labels, 
                    model.sig_length: sig_length, model.base_length: base_length, model.dropout_keep: 1.0, model.is_training:False})
            except tf.errors.OutOfRangeError:
                    print 'Finished Making Predictions'
                    break

def main():
    model = None
    if config.model == 'Baseline':
        model = models.Baseline()
    if config.model == 'SuperBaseline':
        model = super_baseline.SuperBaseline()
    if config.train:
        model.build_graph()
        train(model)
    else: #If we only care about predicting!
        model.build_graph()
        pred(model)

if __name__ == '__main__':
    main()