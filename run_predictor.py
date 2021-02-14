import argparse
import logging as logger

from src import *
from src.runs.predictor import run_predictor
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for evaluation.')
parser.add_argument('--log_to_file',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--model_ckpt',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--dataset',
                    default='genia',
                    type=str,
                    action='store')
parser.add_argument('--device',
                    default=None,
                    type=none_or_str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

config = Config(**args.__dict__)

run_predictor(logger, config)
