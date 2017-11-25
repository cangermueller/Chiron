import csv
import config
from glob import glob
import os
import errno

def baseIndex(base):
    bases = ['a', 'c', 't', 'g']
    return bases.index(base.lower())

def write_labels(label_files):
    strands = []
    base_names = []
    for f in label_files:
        current_strand = ''
        labels = []
        base_name = os.path.basename(f)[:-5] + 'signal'
        with open(f, 'r') as f_l:
            for line in f_l:
                labels += [[x for x in line.strip().split()]] #Start, End, Base
            for trip in labels:
                current_strand += str(baseIndex(trip[2]))
        strands.append(current_strand)
        base_names.append(base_name)
    with open(config.test_label_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(base_names, strands))

def main():
    # Create path to gold labels if necessary
    if not os.path.exists(os.path.dirname(config.gold_labels_file)):
        try:
            os.makedirs(os.path.dirname(config.gold_labels_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

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
    if config.large:
        label_files = ecoli[:2000] + phage[:2000] #This this to get a larger subset of the data
    else:
        label_files = sorted(label_files)
        label_files = label_files[:10] #I am just using a subset of the dataset for experimentation
    write_labels(label_files)

main()
