from glob import glob
import os
import time 
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset, Iterator
import numpy as np
import h5py

import config
import seq2seqModels
import chironModels
import baselineModels
from utils import Database, batch2sparse, sparse2dense, process_labels, yianniSparse2Dense

def train(train_model, val_model):
    train_database = Database(config.train_database)
    val_database = Database(config.val_database)
    best_val_loss = 1000.0
    train_graph = tf.Graph()
    val_graph = tf.Graph()
    with train_graph.as_default():
        train_model.build_train_graph()
        glob_initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver(max_to_keep=1)
    with val_graph.as_default():
        val_model.build_val_graph()
        val_saver = tf.train.Saver()
    train_sess = tf.Session(graph=train_graph)
    val_sess = tf.Session(graph=val_graph)
    train_sess.run(glob_initializer)
    train_writer = tf.summary.FileWriter(os.path.join(config.save_dir, config.model, config.experiment, 'train'), train_sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(config.save_dir, config.model, config.experiment, 'val'), val_sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment, 'val') + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path and not config.restart:
        train_saver.restore(train_sess, ckpt.model_checkpoint_path)
        print 'Restored model from folder ' + ckpt.model_checkpoint_path
    initial_step = train_model.global_step.eval(session=train_sess)
    for i in range(initial_step, config.max_step):
            batch = train_database.get_batch()
            if batch is not None:
                tic = time.clock()
                signals, labels, sig_length, base_length = batch
                if 'Chiron' in config.model or 'Baseline' in config.model:
                    processed_labels = process_labels(labels, base_length)
                    indices, values, shape = batch2sparse(processed_labels)
                    _, loss_batch, train_summary = train_sess.run([train_model.opt, train_model.loss, train_model.summary_op], 
                    feed_dict={train_model.signals: signals, train_model.y_indices:indices, train_model.y_values:values, train_model.y_shape:shape, train_model.sig_length: sig_length, 
                        train_model.base_length: base_length, train_model.dropout_keep: config.dropout_keep, train_model.is_training:True, train_model.lr:config.lr})
                    #predictions = sparse2dense(predictions)[0]
                else:
                    _, loss_batch, train_summary = train_sess.run([train_model.opt, train_model.loss, train_model.summary_op], 
                        feed_dict={train_model.signals: signals, train_model.labels: labels, train_model.sig_length: sig_length, 
                            train_model.base_length: base_length, train_model.dropout_keep: config.dropout_keep, train_model.is_training:True, train_model.lr:config.lr})
                toc = time.clock()
                if config.verbose or i % config.print_every == 0:
                    print 'Batch Loss is ', loss_batch 
                    print str(toc - tic) + ' seconds per batch.'
                #     print predictions[0]
                #     print labels[0]
                train_writer.add_summary(train_summary, global_step=i)
                
                if i % config.val_every == 0: #Perform a validation every config.val_every batches
                    save_path = train_saver.save(train_sess, os.path.join(config.save_dir, config.model, config.experiment, 'train', str(i)) + '.ckpt')
                    val_saver.restore(val_sess, save_path)
                    batch = val_database.get_batch()
                    if batch is not None:
                        signals, labels, sig_length, base_length = batch
                        if 'Chiron' in config.model or 'Baseline':
                            processed_labels = process_labels(labels, base_length)
                            indices, values, shape = batch2sparse(processed_labels)
                            val_loss_batch, val_summary, predictions = val_sess.run([val_model.loss, val_model.summary_op, val_model.predictions], 
                                feed_dict={val_model.signals: signals, val_model.y_indices:indices, val_model.y_values:values, val_model.y_shape:shape, val_model.sig_length: sig_length, 
                                val_model.base_length: base_length, val_model.dropout_keep: 1.0, val_model.is_training:False})
                            predictions = sparse2dense(predictions)[0]
                        else:
                            val_loss_batch, val_summary, predictions = val_sess.run([val_model.loss, val_model.summary_op, val_model.predictions], 
                                feed_dict={val_model.signals: signals, val_model.labels: labels, val_model.sig_length: sig_length, 
                                val_model.base_length: base_length, val_model.dropout_keep: 1.0, val_model.is_training:False})
                        val_writer.add_summary(val_summary, global_step=i)
                        # if config.verbose:
                            # print predictions[0]
                            # print labels[0]
                        if val_loss_batch < best_val_loss:
                            save_path = val_saver.save(val_sess, os.path.join(config.save_dir, config.model, config.experiment, 'val', str(i)) + '.ckpt')
                            print("Model saved in file: %s" % save_path)
                            best_val_loss = val_loss_batch
                        print 'Val Batch Loss is on step ' + str(i), val_loss_batch
                        print 'Best Val Batch Loss is', best_val_loss
                    else:
                        # val_database.reshuffle()
                        print 'No more batches in validation set, wait until next validation interval'      
            else:
                # train_database.reshuffle()
                print 'Ran out of examples on iteration ' + str(i) + ', reinitializing training examples'
    train_writer.close()
    val_writer.close()
    train_database.close()
    val_database.close()

def pred(model):
    model.build_val_graph()
    test_database = Database(config.test_database)
    pred_database = h5py.File(config.predictions_database, 'w')
    pred_database.create_dataset('predicted_bases', (test_database.data['signals'].shape[0],), dtype='S300')
    pred_database.create_dataset('file_names', (test_database.data['signals'].shape[0],), dtype='S200') #Hopefully files aren't > 200 characters...
    pred_database['file_names'][:] = test_database.data['file_names']
    if test_database.data['signals'].shape[0]  % config.batch != 0:
        print 'If signals is not divisible by batch size, you are going to have a bad time.'
        print test_database.data['signals'].shape[0]
        return
    saver = tf.train.Saver()
    predictions = []
    #print len(test_database.data['signals'])
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config.save_dir, config.model, config.experiment, 'val') + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Restored model from folder ' + ckpt.model_checkpoint_path
        counter = 0
        while True:
            batch = test_database.get_batch()
            if counter % 100 == 0:
                print counter
            if batch is not None:
                signals, _, sig_length, _ = batch
                if len(signals) != config.batch: #We really need exactly batch number of samples
                    print 'One of the batches was not batch size'
                    continue
                if 'Chiron' in config.model or 'Baseline':
                    batch_predictions = sess.run([model.predictions], 
                        feed_dict={model.signals: signals, model.sig_length: sig_length, model.dropout_keep: 1.0, model.is_training:False})[0]
                    #batch_predictions1 = sparse2dense(batch_predictions)[0]
                    #batch_predictions2 = [''.join(str(s) for s in pred.tolist()) for pred in batch_predictions1[0]]
                    #print batch_predictions
                    batch_predictions = yianniSparse2Dense(batch_predictions[0])
                    if len(batch_predictions) != 291:
                        # print 'adfasdf'
                        # print batch_predictions
                        print len(batch_predictions)
                        #print batch_predictions[0]
                        #print batch_predictions1
                        #print batch_predictions2
                        #print len(batch_predictions1[0])
                        #print len(batch_predictions2)
                else:
                    batch_predictions = sess.run([model.predictions], 
                        feed_dict={model.signals: signals, model.sig_length: sig_length, model.dropout_keep: 1.0, model.is_training:False})[0].tolist()
                    batch_predictions = [''.join(str(s) for s in pred) for pred in batch_predictions]
                predictions += batch_predictions
                counter += 1
            else:
                print 'All done predicting'
                break
    #print predictions
    #print len(predictions)
    pred_database['predicted_bases'][:] = np.asarray(predictions, dtype=np.string_)
    test_database.close()
    pred_database.close()

def main():
    train_model = None
    val_model = None
    if config.model == 'BabyAchilles':
        train_model = seq2seqModels.BabyAchilles()
        val_model = seq2seqModels.BabyAchilles()
    if config.model == 'Achilles':
        train_model = seq2seqModels.Achilles()
        val_model = seq2seqModels.Achilles()
    if config.model == 'DistractedBabyAchilles':
        train_model = seq2seqModels.DistractedBabyAchilles()
        val_model = seq2seqModels.DistractedBabyAchilles()
    if config.model == 'Chiron_3':
        train_model = chironModels.Chiron_3()
        val_model = chironModels.Chiron_3()
    if config.model == 'Chiron_5':
        train_model = chironModels.Chiron_5()
        val_model = chironModels.Chiron_5()
    if config.model == 'Baseline':
        train_model = baselineModels.Baseline()
        val_model = baselineModels.Baseline()
    if config.train:
        train(train_model, val_model)
    else:
        pred(val_model)

if __name__ == '__main__':
    main()
