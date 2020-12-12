import argparse
import logging as logger

from src.runs.preprocess_genia import run_preprocess_genia
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

parser = argparse.ArgumentParser(description='Arguments for preprocess.')
parser.add_argument('--dataset',
                    type=str,
                    default='genia',
                    action='store',)
parser.add_argument('--cased',
                    type=int,
                    default=False,
                    action='store',)
parser.add_argument('--lm_name',
                    default='dmis-lab/biobert-v1.1',
                    type=str,
                    action='store',)
args = parser.parse_args()

if args.dataset == 'genia':
    run_preprocess_genia(logger, args.cased, args.lm_name)
else:
    logger.error('Unknown dataset: %s' % args.dataset)
    raise Exception('Unknown dataset: %s' % args.dataset)