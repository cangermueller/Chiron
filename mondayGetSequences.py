import csv
import config
from glob import glob
import os
import h5py

def toBase(strand):
    s = ''
    for c in strand:
        if c == '0':
            s += 'A'
        elif c == '1':
            s += 'T'
        elif c == '2':
            s += 'C'
        elif c == '3':
            s += 'G'
        # elif c == '4':
        #     break
    return s

def test(file):
    test_database = h5py.File(config.test_database, 'r')
    current_file = test_database['file_names'][0]
    current_pred_strand, total_pred_strand = '', ''
    print test_database['labels'].shape[0]
    for i in range(test_database['labels'].shape[0]):
        if current_file == test_database['file_names'][i]: #We are still on the same read
            current_pred_strand = ''.join([str(j) for j in list(test_database['labels'][i])])
            # print current_pred_strand
            # print type(current_pred_strand)
            total_pred_strand += current_pred_strand
        else: 
            with open(config.test_label_folder + '/' + os.path.basename(current_file)[:-6] + 'fa', 'w') as f:
                f.write('>' + 'dog' + '\n')
                f.write(toBase(total_pred_strand))
            current_pred_strand = ''.join([str(j) for j in list(test_database['labels'][i])])
            total_pred_strand = current_pred_strand
            current_file = test_database['file_names'][i]
    with open(config.test_label_folder + '/' + os.path.basename(current_file)[:-6] + 'fa', 'w') as f:
        f.write('>' + 'dog' + '\n')
        f.write(toBase(total_pred_strand))


def write_labels(label_files):
    strands = []
    base_names = []
    for f in label_files:
        current_strand = ''
        labels = []
        base_name = os.path.basename(f)[:-5] + 'fa'
        with open(f, 'r') as f_l:
            for line in f_l:
                labels += [[x for x in line.strip().split()]] #Start, End, Base
        for trip in labels:
            current_strand += str(trip[2])
        strands.append(current_strand)
        base_names.append(base_name)
    for pair in zip(base_names, strands):
        with open(config.test_label_folder + '/' + pair[0], 'w') as f:
            f.write('>' + 'dog' + '\n')
            f.write(pair[1])

def main():
    label_files = glob(os.path.join(config.data_read_dir, '*.label'))
    ecoli, phage, human = [], [], []
    for s in label_files:
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
    label_files = ecoli[:2000] + phage[:2000] #This this to get a larger subset of the data
    # write_labels(label_files)
    test(label_files)

main()
