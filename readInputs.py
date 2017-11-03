import numpy as np
from glob import glob
import os
import tensorflow as tf
import config

data_read_dir = config.data_read_dir
data_write_dir = config.data_write_dir

SIG_PAD = 0.0
LAB_PAD = 4
LAB_START = 5

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
    cur_sig, cur_lab, div_sig, div_lab = [], [], [], []
    i = 0 #increment by whenever we actually consume a label triple
    while i < len(all_labels): #Loop through label triples and construct 300 base long sequences
        start, end, base = all_labels[i]
        start = int(start)
        end = int(end)
        if end-start >= config.max_seq_len and len(cur_sig) == 0: #Very rare special case when 1 base is more than max_seq_len signals and there is no cur_sig
            cur_sig = np.asarray(all_signals[start:start+config.max_seq_len-1]) #I'm leaving 1 space for the padding token!
            cur_lab = [baseIndex(base)] + [LAB_PAD]
            cur_sig = (cur_sig - np.mean(cur_sig)) / np.float(np.std(cur_sig))
            cur_sig = cur_sig.tolist() + [SIG_PAD]
            div_sig.append(cur_sig)
            div_lab.append(cur_lab)
            cur_sig, cur_lab = [], []
            i += 1
        elif end - start + len(cur_sig) >= config.max_seq_len: #Done lengthening current sequence
            cur_sig = (cur_sig - np.mean(cur_sig))/np.float(np.std(cur_sig))
            cur_sig = cur_sig.tolist() + [SIG_PAD]
            div_sig.append(cur_sig)
            div_lab.append(cur_lab + [LAB_PAD])
            cur_sig, cur_lab = [], [] #no label triple actually consumed here!
        else:
            cur_sig += all_signals[start:end]
            cur_lab += [baseIndex(base)]
            i += 1
    if cur_sig != []: #If we left any stragglers...
        cur_sig = (cur_sig - np.mean(cur_sig))/np.float(np.std(cur_sig))
        cur_sig = cur_sig.tolist() + [SIG_PAD]
        div_sig.append(cur_sig)
        div_lab.append(cur_lab + [LAB_PAD])
    return div_sig, div_lab

def readAllFiles(signal_files):
    for i in range(len(signal_files)):
        label_file = signal_files[i][:-7] + '.label'
        val = readPair(signal_files[i], label_file)
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
    signal_files = glob(os.path.join(data_read_dir, '*.signal'))
    signal_files = signal_files[:10] #I am just using a subset of the dataset for experimentation
    readAllFiles(signal_files)

main()