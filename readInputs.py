import numpy as np
from glob import glob
import os
import tensorflow as tf
import config

#TO DO: FIX BUG WHERE 1 or 2 BASES ARE NOT PUT ON AT THE END, NOTHING MAJOR RIGHT NOW

data_read_dir = config.data_read_dir
data_write_dir = config.data_write_dir

def baseIndex(base):
    bases = ['a', 'c', 't', 'g']
    return bases.index(base.lower())

def padded(curSeq, curLab):
    curSeq = np.asarray(curSeq)
    curSeq = np.pad(curSeq, pad_width=(0, config.max_seq_len - len(curSeq)), mode='constant', constant_values=0) #Pad input sequence to max length

    curLab = np.asarray(curLab)
    curLab = np.pad(curLab, pad_width=(0, config.max_base_len - len(curLab)), mode='constant', constant_values=4) #Pad base sequence to max length
    return curSeq.tolist(), curLab.tolist()


def readPair(sig_file, lab_file):
    all_signals = list()
    all_labels = list()
    with open(sig_file, 'r') as f_s:
        for line in f_s:
            all_signals += [int(x) for x in line.strip().split()]
        if len(all_signals) == 0:
            return -1
    with open(lab_file, 'r') as f_l:
        all_labels = []
        for line in f_l:
            all_labels += [[x for x in line.strip().split()]] #Start, End, Base
        if len(all_labels) == 0:
            return -1
    curSeq, curLab, allSeq, allLab, sig_length, base_length = list(), list(), list(), list(), list(), list()
    i = 0
    while i < len(all_labels): #Loop through label triples and construct 300 base long sequences
        start, end, base = all_labels[i]
        start = int(start)
        end = int(end)
        if end-start >= config.max_seq_len and len(curSeq) == 0: #Very rare special case when 1 base is more than max_seq_len signals and there is no curSeq
            sig_length.append(config.max_seq_len)
            base_length.append(1)
            curSeq = all_signals[start:start+config.max_seq_len]
            curLab = [baseIndex(base)]
            curSeq, curLab = padded(curSeq, curLab)
            allSeq.append(curSeq)
            allLab.append(curLab)
            curSeq, curLab = list(), list()
            i += 1
        elif end - start + len(curSeq) >= config.max_seq_len: #Done lengthening current sequence
            sig_length.append(len(curSeq))
            base_length.append(len(curLab))
            curSeq, curLab = padded(curSeq, curLab)
            allSeq.append(curSeq)
            allLab.append(curLab)
            curSeq, curLab = list(), list()
        else:
            curSeq += all_signals[start:end]
            curLab += [baseIndex(base)]
            i += 1
    return allSeq, allLab, sig_length, base_length

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def readAllFiles(signal_files, label_files):
    for i in range(len(signal_files)): #Assumes signal and label files exactly line up, may need to sort before to ensure
        val = readPair(signal_files[i], label_files[i])
        if val == -1:
            continue
        allSeq, allLab, sig_length, base_length = val
        for j in range(len(allSeq)):
            file_only = os.path.basename(signal_files[i])
            writer = tf.python_io.TFRecordWriter(os.path.join(data_write_dir, file_only[:-7]) + '_' + str(j) + '.tfrecord') #get rid of .signal ending and groupy shorter runs
            ex = tf.train.SequenceExample()
            ex.context.feature['sig_length'].int64_list.value.append(sig_length[j])
            ex.context.feature['base_length'].int64_list.value.append(base_length[j])
            tokens = ex.feature_lists.feature_list['tokens']
            labels = ex.feature_lists.feature_list['labels']
            for sig in allSeq[j]:
                tokens.feature.add().int64_list.value.append(sig)
            for l in allLab[j]:
                labels.feature.add().int64_list.value.append(l)
            writer.write(ex.SerializeToString())
            writer.close()

def sanityCheck():
    pass

def main():
    signal_files = glob(os.path.join(data_read_dir, '*.signal'))
    label_files = glob(os.path.join(data_read_dir, '*.label'))
    signal_files = signal_files[:10]
    label_files = label_files[:10]
    readAllFiles(signal_files, label_files)

main()