import argparse
import logging as logger

from src import *
from src.runs.training import run_training
from src.utils.config import Config

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--log_to_file',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--model_ckpt',
                    type=none_or_str,
                    action='store')
parser.add_argument('--wv_file',
                    default=None,
                    type=none_or_str,
                    action='store')
parser.add_argument('--use_char_encoder',
                    default=1,
                    type=int,
                    action='store')
parser.add_argument('--dataset',
                    type=str,
                    action='store')
parser.add_argument('--total_layers',
                    default=16,
                    type=int,
                    action='store')
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    action='store')
parser.add_argument('--max_steps',
                    default=int(1e9),
                    type=int,
                    action='store')
parser.add_argument('--max_epoches',
                    default=100,
                    type=int,
                    action='store')
parser.add_argument('--token_emb_dim',
                    default=100,
                    type=int,
                    action='store')
parser.add_argument('--char_emb_dim',
                    default=100,
                    type=int,
                    action='store')
parser.add_argument('--cased_lm',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--cased_word',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--cased_char',
                    default=False,
                    type=int,
                    action='store')
parser.add_argument('--hidden_dim',
                    default=100,
                    type=int,
                    action='store')
parser.add_argument('--dropout',
                    default=0.4,
                    type=float,
                    action='store')
parser.add_argument('--freeze_wv',
                    default=True,
                    type=int,
                    action='store')
parser.add_argument('--lm_name',
                    default='dmis-lab/biobert-v1.1',
                    type=str,
                    action='store')
parser.add_argument('--lm_emb_dim',
                    default=768,
                    type=int,
                    action='store')
parser.add_argument('--continue_training',
                    default=True,
                    type=int,
                    action='store')
parser.add_argument('--device',
                    type=str,
                    action='store')
args = parser.parse_args()
setup_logger(logger, args.log_to_file)

config = Config(**args.__dict__)

run_training(logger, config)
