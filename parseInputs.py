import numpy as np
from glob import glob
import os
import tensorflow as tf
import config
import h5py

data_read_dir = config.data_read_dir

LAB_PAD = 4
LAB_START = 5

def baseIndex(base):
    bases = ['a', 'c', 't', 'g']
    if base.lower() in bases:
        return bases.index(base.lower())
    else:
        return -1

def pad(sequence, pad_token, max_len):
    length = len(sequence)
    if length > max_len:
        sequence = sequence[:max_len]
    else:
        sequence += [pad_token for i in range(max_len - length)]
    return sequence
    
def readPair(sig_file, lab_file):
    signals, labels = [], []
    with open(sig_file, 'r') as f_s:
        for line in f_s:
            signals += [int(x) for x in line.strip().split()]
        if len(signals) == 0:
            return -1
    with open(lab_file, 'r') as f_l:
        for line in f_l:
            labels += [[x for x in line.strip().split()]] #Start, End, Base
        if len(labels) == 0:
            return -1
    if config.normalize:
        signals = np.asarray(signals) #Normalize the entire sequence BEFORE splitting
        signals = (signals - np.mean(signals)) / np.float(np.std(signals))
        signals = signals.tolist()
    cur_sig, cur_lab, split_sig, split_lab, sig_len, base_len = [], [], [], [], [], []
    for i in range(len(labels)):
        start, end, base = labels[i]
        start = int(start)
        end = int(end)
        if end - start + len(cur_sig) < config.max_seq_len: #We may have a complete example, pending quality control
            cur_sig += signals[start:end]
            cur_lab += [baseIndex(base)]
        else:
            if len(cur_sig) > config.max_seq_len / 2 and len(cur_lab) > 5:
                sig_len.append(len(cur_sig))
                base_len.append(len(cur_lab) + 1) #Include the end token too
                split_sig.append(pad(cur_sig, 0.0, config.max_seq_len))
                split_lab.append(pad(cur_lab + [LAB_PAD], LAB_PAD, config.max_base_len))
            cur_sig = signals[start:end]
            cur_lab = [baseIndex(base)]
            
    return split_sig, split_lab, sig_len, base_len

def readAllFiles(signal_files):
    problemFiles = []
    f = h5py.File(config.write_database, 'w')
    dset1 = f.create_dataset('signals', (0, config.max_seq_len), maxshape=(None, config.max_seq_len), dtype='float32')
    dset2 = f.create_dataset('labels', (0, config.max_base_len), maxshape=(None, config.max_base_len), dtype='int32')
    dset3 = f.create_dataset('signal_lengths', (0,), maxshape=(None,), dtype='int32')
    dset4 = f.create_dataset('base_lengths', (0,), maxshape=(None,), dtype='int32')
    dset5 = f.create_dataset('file_names', (0,), maxshape=(None,), dtype='S200')

    for i in range(len(signal_files)):
        if i % 1000 == 0:
            print 'On file ' + str(i)
        base_name = os.path.basename(signal_files[i])
        label_file = os.path.splitext(signal_files[i])[0] + '.label'
        output = readPair(signal_files[i], label_file)
        if output == -1:
            problemFiles.append(label_file)
            continue
        split_sig, split_lab, sig_len, base_len = output
        cur_size = dset1.shape[0]
        dset1.resize(cur_size + len(split_sig), axis=0)
        dset2.resize(cur_size + len(split_sig), axis=0)   
        dset3.resize(cur_size + len(split_sig), axis=0)   
        dset4.resize(cur_size + len(split_sig), axis=0) 
        dset5.resize(cur_size + len(split_sig), axis=0)
        dset1[-len(split_sig):, :] = np.asarray(split_sig, dtype=np.float32)
        dset2[-len(split_sig):, :] = np.asarray(split_lab, dtype=np.int32)
        dset3[-len(split_sig):] = np.asarray(sig_len, dtype=np.int32) 
        dset4[-len(split_sig):] = np.asarray(base_len, dtype=np.int32) 
        dset5[-len(split_sig):] = np.asarray([base_name for i in range(len(split_sig))], dtype=np.string_)
    print str(dset1.shape[0]) + ' examples created.'
    f.close()
    print problemFiles

def main():
    signal_files = glob(os.path.join(data_read_dir, '*.signal'))
    ecoli, phage, human = [], [], []
    for s in signal_files:
        sl = s.lower()
        if 'lambda' in sl:
            phage.append(s)
        elif 'human' in sl:
            human.append(s)
        else:
            ecoli.append(s)
    ecoli = sorted(ecoli)
    phage = sorted(phage)
    human = sorted(human)
    print len(ecoli), len(phage), len(human)
    if config.large:
        signal_files = ecoli[:2000] + phage[:2000] #This this to get a larger subset of the data
    else:
        signal_files = sorted(signal_files)
        signal_files = signal_files[:10] #I am just using a subset of the dataset for experimentation
    readAllFiles(signal_files)

main()