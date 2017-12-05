import numpy as np
import h5py
from glob import glob
import os
import config

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
        elif c == '4':
            break
    return s

def process_predictions():
    pred_database = h5py.File(config.predictions_database, 'r')
    current_pred_strand, total_pred_strand = '', ''
    current_file = pred_database['file_names'][0]
    if not os.path.exists(config.predictions_fasta_folder):
        os.makedirs(config.predictions_fasta_folder)
    for i in range(pred_database['predicted_bases'].shape[0]):
        if current_file == pred_database['file_names'][i]: #We are still on the same read
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand += current_pred_strand
        else: #We are done predicting a given run...
            with open(config.predictions_fasta_folder + '/' + os.path.basename(current_file)[:-6] + 'fasta', 'w') as f:
                f.write('>' + 'dog' + '\n')
                f.write(toBase(total_pred_strand))
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand = current_pred_strand
            current_file = pred_database['file_names'][i]
    with open(config.predictions_fasta_folder + '/' + os.path.basename(current_file)[:-6] + 'fasta', 'w') as f: #get the last 1 too
        f.write('>' + 'dog' + '\n')
        f.write(toBase(total_pred_strand))

def main():
    process_predictions()

main()