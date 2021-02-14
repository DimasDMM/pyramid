import os
from transformers import AutoTokenizer, AutoConfig

from .. import *

def get_tokenizer(artifacts_path='artifacts/', lm_name='bert-base-multilingual-cased',
                  lowercase=True, max_length=512):
    save_path = get_project_path(artifacts_path, lm_name)
    tokenizer = AutoTokenizer.from_pretrained(save_path, lowercase=lowercase, use_fast=True, model_max_length=max_length,
                                              config=AutoConfig.from_pretrained(os.path.join(save_path, 'config.json')))
    return tokenizer
