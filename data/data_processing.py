"""
Data processing file
1. divide dataset ：
    python data_processing.py --div DEYN --div ./demo/
data 2019.1.10
author zh lee
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging


def _parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser('Pre_procession of raw data set')
    parser.add_argument('--div', choices=['DEYN', 'FO'], default='DEYN',
                                help="if divide data set into sub set, "+
                                     "choose DEYN: divide data use Answer type: Description / Entity / Yes_No or"+
                                     "choose FO: divide data use Question type: Fact / Opinion")
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--div_dir', default='./demo/',
                               help='the dir with data set preprocessed to be divide')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def run():
    _parse_args()
    logger = logging.getLogger("data_process")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

if __name__ == '__main__':
    run()