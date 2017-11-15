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

def getDatasetIterator(sig, padding_num=None):
    sig_files = glob(sig)
    lab_files = [f[:-11] + '_label.txt' for f in sig_files]
    sig_dataset = tf.contrib.data.TextLineDataset(sig_files)
    sig_dataset = sig_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    sig_dataset = sig_dataset.map(lambda sequence: (sequence, tf.size(sequence))) #Dont forget about this!!!
    sig_dataset = sig_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.float32), length))
    if config.train == False: #We only have signal files when predicting
        batched_dataset = sig_dataset.padded_batch(
            config.batch,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=((0.0, 0)))
        batched_iterator = batched_dataset.make_initializable_iterator()
        return batched_iterator, sig_files
    lab_dataset = tf.contrib.data.TextLineDataset(lab_files)
    lab_dataset = lab_dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
    lab_dataset = lab_dataset.map(lambda sequence: (sequence, tf.size(sequence)))
    lab_dataset = lab_dataset.map(lambda seq, length: (tf.string_to_number(seq, out_type=tf.int32), length))
    dataset = tf.contrib.data.Dataset.zip((sig_dataset, lab_dataset))
    batched_shuffled_dataset = dataset.shuffle(config.batch)
    batched_shuffled_padded_dataset = batched_shuffled_dataset.padded_batch(
        config.batch,
        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                        tf.TensorShape([])),     # size(source)
                       (tf.TensorShape([padding_num]),  # target vectors of unknown size
                        tf.TensorShape([]))),    # size(target)
        padding_values=((0.0, 0), (4, 0)))       #Pad signal with 0.0 and label with 4 and don't pad the two lengths
    batched_iterator = batched_shuffled_padded_dataset.make_initializable_iterator()   
    return batched_iterator

def train(train_model, val_model):
    best_val_loss = 1000.0
    train_graph = tf.Graph()
    val_graph = tf.Graph()
    with train_graph.as_default():
        train_model.build_train_graph()
        glob_initializer = tf.global_variables_initializer()
        loc_initializer = tf.local_variables_initializer()
        train_iterator = getDatasetIterator(os.path.join(train_dir, '*signal.txt'), padding_num=None)
        train_batch = train_iterator.get_next()
        train_saver = tf.train.Saver(max_to_keep=1)
    with val_graph.as_default():
        val_model.build_val_graph()
        val_loc_initializer = tf.local_variables_initializer()
        val_iterator = getDatasetIterator(os.path.join(val_dir, '*signal.txt'), padding_num=config.max_base_len)
        val_batch = val_iterator.get_next()
        val_saver = tf.train.Saver()
    
    train_sess = tf.Session(graph=train_graph)
    val_sess = tf.Session(graph=val_graph)
    train_sess.run(glob_initializer)
    train_sess.run(loc_initializer)
    train_sess.run(train_iterator.initializer)
    val_sess.run(val_iterator.initializer)
    val_sess.run(val_loc_initializer)

    train_writer = tf.summary.FileWriter(os.path.join(config.save_dir, config.model, config.experiment, 'train'), train_sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(config.save_dir, config.model, config.experiment, 'val'), val_sess.graph)
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment, 'val') + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path and not config.restart:
        train_saver.restore(train_sess, ckpt.model_checkpoint_path)
        print 'Restored model from folder ' + ckpt.model_checkpoint_path
    initial_step = train_model.global_step.eval(session=train_sess)

    for i in range(initial_step, config.max_step):
            try:
                batch = train_sess.run(train_batch)
                signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 
                if len(signals) != config.batch: #We need exactly batch number of examples (no remainders)
                    continue
                _, loss_batch, train_summary = train_sess.run([train_model.opt, train_model.loss, train_model.summary_op], 
                    feed_dict={train_model.signals: signals, train_model.labels: labels, train_model.sig_length: sig_length, 
                        train_model.base_length: base_length, train_model.dropout_keep: config.dropout_keep, train_model.is_training:True, train_model.lr:config.lr})
                if config.verbose or config.print_every % i == 0:
                    print 'Batch Loss is ', loss_batch 
                train_writer.add_summary(train_summary, global_step=i)
                
                if i % config.val_every == 0: #Perform a validation every config.val_every batches
                    save_path = train_saver.save(train_sess, os.path.join(config.save_dir, config.model, config.experiment, 'train', str(i)) + '.ckpt')
                    val_saver.restore(val_sess, save_path)
                    try:
                        batch = val_sess.run(val_batch)
                        signals, labels, sig_length, base_length = batch[0][0], batch[1][0], batch[0][1], batch[1][1] 
                        
                        if len(signals) != config.batch: #We need exactly batch number of examples (no remainders)
                            continue
                        val_loss_batch, val_summary = val_sess.run([val_model.loss, val_model.summary_op], 
                            feed_dict={val_model.signals: signals, val_model.labels: labels, val_model.sig_length: sig_length, 
                            val_model.base_length: base_length, val_model.dropout_keep: 1.0, val_model.is_training:False})
                        val_writer.add_summary(val_summary, global_step=i)
                        if val_loss_batch < best_val_loss:
                            save_path = val_saver.save(val_sess, os.path.join(config.save_dir, config.model, config.experiment, 'val', str(i)) + '.ckpt')
                            print("Model saved in file: %s" % save_path)
                            best_val_loss = val_loss_batch
                        print 'Val Batch Loss is on step ' + str(i), val_loss_batch
                        print 'Best Val Batch Loss is', best_val_loss
                    except tf.errors.OutOfRangeError:
                        val_sess.run(val_iterator.initializer)
                        print 'No more batches in validation set, wait until next validation interval'
            
            except tf.errors.OutOfRangeError:
                train_sess.run(train_iterator.initializer)
                print 'Ran out of examples on iteration ' + str(i) + ', reinitializing training examples'
    train_writer.close()
    val_writer.close()

def pred(model):
    model.build_val_graph()
    pred_iterator, pred_files = getDatasetIterator(os.path.join(config.pred_dir, '*signal.txt'))
    if len(pred_files) % config.batch != 0:
        print 'If signals is not divisible by batch size, you are going to have a bad time.'
        print len(pred_files)
        return
    pred_batch = pred_iterator.get_next()
    saver = tf.train.Saver()
    predictions = []
    print pred_files[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(pred_iterator.initializer)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment, 'val') + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path
        while True:
            try:
                batch = sess.run(pred_batch)
                signals, sig_length = batch
                if len(signals) != config.batch: #We really need exactly batch number of samples
                    continue
                batch_predictions = sess.run([model.predictions], feed_dict={model.signals: signals, model.sig_length: sig_length, 
                    model.dropout_keep: 1.0, model.is_training:False})[0]
                predictions += batch_predictions.tolist()
            except tf.errors.OutOfRangeError:
                    print 'Finished Making Predictions'
                    break
    for i in range(len(pred_files)):
        with open(os.path.join(config.pred_output_dir, os.path.basename(pred_files[i])), 'w') as f:
            f.write(str(predictions[i])[1: -1])

def main():
    train_model = None
    val_model = None
    if config.model == 'BabyAchilles':
        train_model = models.BabyAchilles()
        val_model = models.BabyAchilles()
    if config.model == 'Achilles':
        train_model = models.Achilles()
        val_model = models.Achilles()
    if config.train:
        train(train_model, val_model)
    else:
        pred(val_model)

if __name__ == '__main__':
    main()