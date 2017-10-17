#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:38:18 2017

@author: haotianteng
"""
import argparse
import logging
import os
import sys

import chiron_eval
import chiron_rcnn_train
from utils.extract_sig_ref import extract
from utils import raw


def evaluation(args, log):
    FLAGS=args
    FLAGS.input_dir=FLAGS.input
    FLAGS.output_dir=FLAGS.output
    extract(FLAGS)
    FLAGS.input = FLAGS.output+'/raw/'
    chiron_eval.run(args)

def export(args, log):
    raw.run(args, log)

def main(argv=sys.argv):
    name = os.path.basename(argv[0])

    parser=argparse.ArgumentParser(prog='chiron',description='A deep neural network basecaller.')
    subparsers = parser.add_subparsers(title='sub command',help='sub command help')
    model_default_path=__file__+'/../model/DNA_default'

    parser.add_argument('-v', '--verbose', help='More detailed logging.', action='store_true')
    parser.add_argument('--log_file', help='Write log messages to file')

    #parser for 'call' command
    parser_call=subparsers.add_parser('call', description='Perform basecalling',help='Perform basecalling.')
    parser_call.add_argument('-i','--input',required=True, help="File path or Folder path to the fast5 file.")
    parser_call.add_argument('-o','--output',required = True, help = "Output folder path")
    parser_call.add_argument('-m','--model', default = model_default_path,help = "model folder")
    parser_call.add_argument('-s','--start',type=int,default = 0,help = "Start index of the signal file.")
    parser_call.add_argument('-b','--batch_size',type = int,default = 100,help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
    parser_call.add_argument('-l','--segment_len',type = int,default = 300, help="Segment length to be divided into.")
    parser_call.add_argument('-j','--jump',type = int,default = 30,help = "Step size for segment")
    parser_call.add_argument('-t','--threads',type = int,default = 0,help = "Threads number")
    parser_call.add_argument('-e','--extension',default = 'fastq',help = "Output file type.")
    parser_call.set_defaults(func=evaluation)

    #parser for 'extract' command
    parser_export=subparsers.add_parser('export',description='Export signal and label from the fast5 file.',help='Extract signal and label in the fast5 file.')
    parser_export.add_argument('-i','--input',required=True,help='Input folder contain fast5 files.')
    parser_export.add_argument('-o','--output',required=True,help='Output folder.')
    parser.add_argument('--basecall_group',default = 'Basecall_1D_000',help = 'Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup',default = 'BaseCalled_template',help = 'Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser_export.set_defaults(func=export)

    #parser for 'train' command
    parser_train=subparsers.add_parser('train',description='Model training',help='Train a model.')
    parser_train.add_argument('-i','--data_dir',required=True,help="Folder containing the labelled data.")
    parser_train.add_argument('-o','--log_dir',required=True,help="Log dir which save the trained model")
    parser_train.add_argument('-c','--cache_dir',required=True,help="Caching directory.")
    parser_train.add_argument('-n','--model_name',required=True,help="Model name saved.")
    parser_train.add_argument('-t','--retrain',type=bool,default=False,help="If retrain is true, the previous trained model will be loaded from LOG_DIR before training.")
    parser_train.add_argument('-l','--sequence_len',type=int,default=300,help="Segment length to be divided into.")
    parser_train.add_argument('-b','--batch_size',type=int,default=300,help="Batch size to train, large batch size require more ram but give faster training speed.")
    parser_train.add_argument('--max_files',type=int,help="Maximum number of input files.")
    parser_train.add_argument('--max_samples',type=int,help="Maximum number of training samples (or reads).")
    parser_train.add_argument('-m','--max_steps',type=int,default=20000,help="Maximum training steps per epoch.")
    parser_train.add_argument('-r','--step_rate',type=float,default=1e-3,help="Learning rate used for optimiztion algorithm.")
    parser_train.add_argument('-k','--k_mer',type=int,default=1,help="Output k-mer size.")
    parser_train.set_defaults(func=chiron_rcnn_train.run)

    args = parser.parse_args(argv[1:])

    logging.basicConfig(filename=args.log_file,
                        format='%(levelname)s (%(asctime)s): %(message)s')
    log = logging.getLogger(name)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    log.info(argv[1:])

    args.func(args, log)

if __name__=='__main__':
    main()
