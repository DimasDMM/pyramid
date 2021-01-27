import argparse
import logging as logger

from src import *
from src.runs.preprocess_genia import run_preprocess_genia
from src.runs.training import run_training
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for preprocess.')
parser.add_argument('--log_to_file',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--dataset',
                    type=str,
                    default='genia',
                    action='store')
parser.add_argument('--raw_filepath',
                    type=str,
                    default='./data/GENIA/GENIAcorpus3.02.merged.xml',
                    action='store')
parser.add_argument('--cased',
                    type=int,
                    default=False,
                    action='store')
parser.add_argument('--lm_name',
                    default='dmis-lab/biobert-v1.1',
                    type=str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

if args.dataset == 'genia':
    run_preprocess_genia(logger, args.raw_filepath, args.cased, args.lm_name)
else:
    logger.error('Unknown dataset: %s' % args.dataset)
    raise Exception('Unknown dataset: %s' % args.dataset)
