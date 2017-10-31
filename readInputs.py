import numpy as np
from glob import glob
import os
import tensorflow as tf
import config

data_read_dir = config.data_read_dir
data_write_dir = config.data_write_dir

def baseIndex(base):
    bases = ['a', 'c', 't', 'g']
    return bases.index(base.lower())

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
    cur_sig, cur_lab, div_sig, div_lab = list(), list(), list(), list()
    cur_lab += [5]
    i = 0 #increment by whenever we actually consume a label triple
    while i < len(all_labels): #Loop through label triples and construct 300 base long sequences
        start, end, base = all_labels[i]
        start = int(start)
        end = int(end)
        if end-start >= config.max_seq_len and len(cur_sig) == 0: #Very rare special case when 1 base is more than max_seq_len signals and there is no cur_sig
            cur_sig = np.asarray(all_signals[start:start+config.max_seq_len-1]) #I'm leaving 1 space for the padding token!
            cur_lab = [baseIndex(base)]
            cur_sig = (cur_sig - np.mean(cur_sig)) / np.float(np.std(cur_sig))
            cur_sig = cur_sig.tolist()
            div_sig.append(cur_sig)
            div_lab.append(cur_lab)
            cur_sig, cur_lab = list(), [5]
            i += 1
        elif end - start + len(cur_sig) >= config.max_seq_len: #Done lengthening current sequence
            cur_sig = (cur_sig - np.mean(cur_sig))/np.float(np.std(cur_sig))
            cur_sig = cur_sig.tolist()
            div_sig.append(cur_sig)
            div_lab.append(cur_lab )
            cur_sig, cur_lab = list(), [5] #no label triple actually consumed here!
        else:
            cur_sig += all_signals[start:end]
            cur_lab += [baseIndex(base)]
            i += 1
    if cur_sig != []: #If we left any stragglers...
        cur_sig = (cur_sig - np.mean(cur_sig))/np.float(np.std(cur_sig))
        cur_sig = cur_sig.tolist()
        div_sig.append(cur_sig)
        div_lab.append(cur_lab)
    return div_sig, div_lab

def readAllFiles(signal_files, label_files):
    for i in range(len(signal_files)):
        val = readPair(signal_files[i], label_files[i])
        if val == -1:
            continue
        div_sig, div_lab = val
        for j in range(len(div_sig)):
            file_only = os.path.basename(signal_files[i])
            with open(os.path.join(data_write_dir, file_only[:-7]) + '_' + str(j) + '_signal.txt', 'w') as f:
                f.write(str(div_sig[j])[1: -1]) #hack to get rid of [ and ] from str(list)
            with open(os.path.join(data_write_dir, file_only[:-7]) + '_' + str(j) + '_label.txt', 'w') as f:
                f.write(str(div_lab[j])[1: -1])


def main():
    signal_files = glob(os.path.join(data_read_dir, '*.signal')) #May want to sort these but it is giving them in pairs for now...
    label_files = glob(os.path.join(data_read_dir, '*.label'))
    signal_files = signal_files[:10] #I am just using a subset of the dataset for experimentation
    label_files = label_files[:10]
    readAllFiles(signal_files, label_files)

main()