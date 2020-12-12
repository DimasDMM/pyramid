import argparse
import logging as logger

from src.runs.training import run_training
from src.utils.config import Config

logger.basicConfig(level=logger.DEBUG)

if __name__ != '__main__':
    exit(0)

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--model_ckpt',
                    default=None,
                    type=none_or_str,
                    action='store',)
parser.add_argument('--wv_file',
                    default='./data/embeddings/glove.6B.100d.txt',
                    type=str,
                    action='store',)
parser.add_argument('--dataset',
                    default='genia',
                    action='store',)
parser.add_argument('--total_layers',
                    default=16,
                    type=int,
                    action='store',)
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    action='store',)
parser.add_argument('--evaluate_interval',
                    default=1000,
                    type=int,
                    action='store',)
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
                    action='store',)
parser.add_argument('--char_emb_dim',
                    default=100,
                    type=int,
                    action='store',)
parser.add_argument('--cased',
                    default=False,
                    type=int,
                    action='store',)
parser.add_argument('--hidden_dim',
                    default=100,
                    type=int,
                    action='store',)
parser.add_argument('--dropout',
                    default=0.4,
                    type=float,
                    action='store',)
parser.add_argument('--freeze_wv',
                    default=True,
                    type=int,
                    action='store',)
parser.add_argument('--lm_name',
                    default='dmis-lab/biobert-v1.1',
                    type=str,
                    action='store',)
parser.add_argument('--lm_emb_dim',
                    default=768,
                    type=int,
                    action='store',)
parser.add_argument('--device',
                    default=None,
                    type=none_or_str,
                    action='store',)

args = parser.parse_args()
config = Config(**args.__dict__)

run_training(logger, config)
