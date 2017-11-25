import numpy as np
import Bio
from Bio import pairwise2
import config
import h5py

def calculate_stats(str1, str2):
    if len(str1) == 0:
        return 0.0, 0.0, 1.0, 0.0, 1.0
    alignment = pairwise2.align.globalms(str1, str2, 2, -.5, -.5, -.1, one_alignment_only=True)
    al1, al2, score, begin, end = alignment[0]
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

def getStatsLineByLine():
    test = h5py.File(config.test_database, 'r')
    pred = h5py.File(config.predictions_database, 'r')
    all_stats = []
    for i in range(len(pred['predicted_bases'])):
        if i % 1000 == 0:
            print i
        basecall = pred['predicted_bases'][i]
        label = test['labels'][i].tolist()[:test['base_lengths'][i]-1]
        label = [str(i) for i in label]
        label = ''.join(label)
        stats = calculate_stats(basecall, label)
        all_stats.append(stats)
    all_stats = np.asarray(all_stats)
    av_stats = np.mean(all_stats, axis=0)
    with open('stats.txt', 'w') as f:
        f.write(str(av_stats.tolist())[1:-1])
    pred.close()
    test.close()

def main():
    getStatsLineByLine()

main()

