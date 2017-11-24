import csv
import config
from glob import glob
import os

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
    write_labels(label_files)

main()
