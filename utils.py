from glob import glob
import os
import time 
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset, Iterator
import numpy as np
import h5py

import config

class Database:
    def __init__(self, f):
        self.data = h5py.File(f, 'r')
        self.batch_num = 0
        # self.ind = np.arange(self.data['base_lengths'].shape[0])
        # np.random.shuffle(self.ind)

    def get_batch(self):
        batch_num = self.batch_num
        if batch_num * config.batch + config.batch <= len(self.data['base_lengths']):
            base_lengths = self.data['base_lengths'][batch_num*config.batch: batch_num*config.batch+config.batch]
            signal_lengths = self.data['signal_lengths'][batch_num*config.batch: batch_num*config.batch+config.batch]
            signals = self.data['signals'][batch_num*config.batch: batch_num*config.batch+config.batch, :]
            labels = self.data['labels'][batch_num*config.batch: batch_num*config.batch+config.batch, :]
            self.batch_num += 1
            return signals, labels, signal_lengths, base_lengths
        else:
            self.batch_num = 0 #We have run out of data!
            return None

    def reshuffle(self):
        self.ind = np.arange(len(self.data['base_lengths']))
        np.random.shuffle(self.ind)

    def close(self):
        self.data.close()

#From chiron_input.py
def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor"""
    values = []
    indices = []
    for batch_i,label_list in enumerate(label_batch[:,0]):
        for indx,label in enumerate(label_list):
            if indx>=label_batch[batch_i,1]:
                break
            indices.append([batch_i,indx])
            values.append(label)
    shape = [len(label_batch),max(label_batch[:,1])]
    return (indices,values,shape)
    
def sparse2dense(predict_val):
    predict_val_top5 = predict_val[0]
    predict_read = list()
    uniq_list = list()
    for i in range(len(predict_val_top5)):
        predict_val = predict_val_top5[i]
        unique,pre_counts = np.unique(predict_val.indices[:,0],return_counts = True)
        uniq_list.append(unique)
        pos_predict = 0
        predict_read_temp = list()
        for indx,counts in enumerate(pre_counts):
            predict_read_temp.append(predict_val.values[pos_predict:pos_predict+pre_counts[indx]])
            pos_predict +=pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read,uniq_list

def process_labels(labels_np, lengths_np):
    lengths_np -= 1 #Get rid of the padding that we used for seq2seq
    lengths = [bl for bl in lengths_np.tolist()]
    labels_list = labels_np.tolist()
    labels_trimmed = [labels_list[i][:lengths[i]] for i in range(len(lengths))]
    return np.asarray(zip(labels_trimmed, lengths))