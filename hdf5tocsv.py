import h5py
import config
import os
import csv
import errno

def getPreds(file):
    pred_database = h5py.File(file, 'r')
    current_file = pred_database['file_names'][0]
    current_strand, total_pred_strand = '', ''
    strand = []
    files = []
    counter = 0
    for i in range(pred_database['predicted_bases'].shape[0]):
        if current_file == pred_database['file_names'][i]: #We are still on the same read
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand += current_pred_strand
        else: #We are done predicting a given run...
            if counter % 100 == 0:
                print counter
            counter += 1
            strand.append(total_pred_strand)
            files.append(current_file)
            current_pred_strand = pred_database['predicted_bases'][i]
            total_pred_strand = current_pred_strand
            current_file = pred_database['file_names'][i]
    strand.append(total_pred_strand)
    files.append(current_file)
    with open(config.pred_labels_file, 'w') as f:
        writer = csv.writer(f)
        print len(zip(files, strand))
        writer.writerows(zip(files, strand))

def main():
    # Create path to prediction labels file if necessary
    if not os.path.exists(os.path.dirname(config.pred_labels_file)):
        try:
            os.makedirs(os.path.dirname(config.pred_labels_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    getPreds(config.predictions_database)

main()
