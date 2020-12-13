import argparse
import logging as logger
import os
from transformers import BertModel
from transformers import BertTokenizer

logger.basicConfig(level=logger.DEBUG)

if __name__ != '__main__':
    exit(0)

parser = argparse.ArgumentParser(description='Arguments for LM Download.')
parser.add_argument('--lm_name',
                    default='dmis-lab/biobert-v1.1',
                    type=str,
                    action='store')
args = parser.parse_args()

logger.info('== LM DOWNLOADER ==')

save_path = './artifacts/%s/' % args.lm_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger.info('Download tokenizer')
slow_tokenizer = BertTokenizer.from_pretrained(args.lm_name)
slow_tokenizer.save_pretrained(save_path)

logger.info('Download model')
model = BertModel.from_pretrained(args.lm_name)
model.save_pretrained(save_path)

logger.info('Done')
