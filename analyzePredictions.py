import subprocess
import glob
import os
import config
import numpy as np

def getStats(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f]
        delRate, insRate, misRate, idRate = 0.0, 0.0, 0.0, 0.0
        for i in range(len(lines)):
            if lines[i][0] == '=': #First line of summary starts with =
                total_reads = lines[i+1].split(':')[1]
                unaligned = lines[i+2].split(':')[1].strip()
                if unaligned == '0': #Should have aligned everything
                    delRate = lines[i+3].split(':')[1]
                    insRate = lines[i+4].split(':')[1]
                    misRate = lines[i+5].split(':')[1]
                    idRate = lines[i+6].split(':')[1]
                break
        return delRate, insRate, misRate, idRate

def main():
    reference_sequences = glob.glob(os.path.join(config.test_label_folder, '*.fa'))
    aligned_stats = []
    unaligned_stats = []
    unaligned = 0
    for i, seq in enumerate(reference_sequences):
        if i % 100 == 0:
            print i
        prediction_file = config.predictions_fasta_folder + '/' + os.path.basename(seq)[:-2] + 'fasta'
        if not os.path.exists(config.predictions_bam_folder):
            os.makedirs(config.predictions_bam_folder)
        alignment_file = config.predictions_bam_folder + '/' + os.path.basename(seq)[:-2] + 'sam'
        with open(alignment_file, 'w') as f:
            subprocess.call(['minimap2', '-a', '-k6', '-w4', seq, prediction_file], stdout=f) #align
        with open(alignment_file[:-3] + 'bam', 'w') as f:
            subprocess.call(['samtools', 'view', '-bS', alignment_file], stdout=f) #convert to bam
        subprocess.call(['rm', alignment_file]) #get rid of sam file but keep bam file
        with open('jsaStats.txt', 'w') as f:
            subprocess.call(['jsa.hts.errorAnalysis', '-reference=' + seq, '-bamFile=' + alignment_file[:-3] + 'bam'], stdout=f)
        delRate, insRate, misRate, idRate = getStats('jsaStats.txt')
        subprocess.call(['rm', 'jsaStats.txt'])
        if idRate == 0.0:
            unaligned += 1
            unaligned_stats.append([delRate, insRate, misRate, idRate])
        else:
            unaligned_stats.append([delRate, insRate, misRate, idRate])
            aligned_stats.append([delRate, insRate, misRate, idRate])
    aligned_stats = np.asarray(aligned_stats, dtype=np.float32)
    unaligned_stats = np.asarray(unaligned_stats, dtype=np.float32)
    av_aligned_stats = np.mean(aligned_stats, axis=0)
    av_unaligned_stats = np.mean(unaligned_stats, axis=0)
    print av_aligned_stats
    print av_unaligned_stats
    print unaligned

main()