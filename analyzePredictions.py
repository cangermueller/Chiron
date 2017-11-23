import numpy as np
import h5py
from glob import glob
import os
import Bio
from Bio import pairwise2
import nltk
from nltk import metrics
from nltk.metrics import edit_distance
import csv
import config

def get_true_strands():
    data = {}
    with open(config.test_label_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data[row[0]] = row[1]
    return data

def calculate_stats(str1, str2):
    reflen = len(str2)
    alignment = pairwise2.align.globalms(str1, str2, 2, -1, -.5, -.1)[0]
    al1, al2, score, begin, end = alignment
    identity, mismatch, deletion, insertion = 0.0, 0.0, 0.0, 0.0
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
    ed = edit_distance(al1, al2)
    return identity / reflen, mismatch / reflen, deletion / reflen, insertion / reflen, 1.0*ed / reflen

def process_predictions(true_strands):
    pred_database = h5py.File(config.predictions_database, 'r')
    current_pred_strand, total_pred_strand = '', ''
    current_file = pred_database['file_names'][0]
    true_strand = true_strands[current_file]
    stats = []
    counter = 0
    for i in range(pred_database['predicted_bases'].shape[0]):
        if current_file == pred_database['file_names'][i]: #We are still on the same read
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand += current_pred_strand
        else: #We are done predicting a given run...
           # print total_pred_strand, true_strand
            if counter % 10 == 0:
                print counter
            print counter
            counter += 1
            identity, mismatch, deletion, insertion, ed = calculate_stats(total_pred_strand, true_strand)
            stats.append([identity, mismatch, deletion, insertion, ed])
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand += current_pred_strand
            if i != pred_database['predicted_bases'].shape[0] - 1:
                current_file = pred_database['file_names'][i+1]
                true_strand = true_strands[current_file]
            #print identity, mismatch, deletion, insertion, ed
    identity, mismatch, deletion, insertion, ed = calculate_stats(current_pred_strand, true_strand)
    stats.append([identity, mismatch, deletion, insertion, ed])
    stats = np.asarray(stats, dtype=float32)
    print len(stats), pred_database['predicted_bases'].shape[0]
    av_stats = np.mean(stats, axis=0)
    return av_stats

def main():
    true_strands = get_true_strands()
    av_stats = process_predictions(true_strands)
    print av_stats

main()
