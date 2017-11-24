import numpy as np
from glob import glob
import os
import Bio
from Bio import pairwise2
import csv

def calculate_stats(str1, str2):
    alignment = pairwise2.align.globalms(str1, str2, 2, -.5, -.5, -.1)[0]
    al1, al2, score, begin, end = alignment
    identity, mismatch, deletion, insertion = 0.0, 0.0, 0.0, 0.0
    reflen = len(al2)
    for i in range(len(al1)):
        if al1[i] != '-' and al2[i] != '-':
            if al1[i] == al2[i]:
                identity += 1.0
            else:
                mismatch += 1.0
        else:
            if al1[i] == '-':
                deletion += 1.0
            elif al2[i] == '-':
                insertion += 1.0
    ed = al1.count('-') + mismatch
    return identity / reflen, mismatch / reflen, deletion / reflen, insertion / reflen, 1.0*ed / reflen

def process_predictions(true, pred):
    print len(true), len(pred)
    stats = []
    for i in range(len(true)):
        if i % 10 == 0:
            print i
        assert true[i][0][:-5] == pred[i][0][:-6]
        identity, mismatch, deletion, insertion, ed = calculate_stats(pred[i][1], true[i][1])
        stats.append([identity, mismatch, deletion, insertion, ed])
    stats = np.asarray(stats, dtype=np.float32)
    print np.mean(stats, axis=0)
    return np.mean(stats, axis=0)

def main():
    true = []
    pred = []
    with open('val_labels.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            true.append(row)
    with open('Chiron_3exp1size200drop80lr001predictions.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            pred.append(row)
    av_stats = process_predictions(true, pred)
    with open('stats.txt', 'w') as f:
        f.write(str(av_stats.tolist())[1:-1])

main()
