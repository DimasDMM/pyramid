import os
from transformers import AutoTokenizer, AutoConfig

from .. import *

def get_tokenizer(artifacts_path='artifacts/', lm_name='bert-base-cased',
                  lowercase=False, max_length=512):
    save_path = get_project_path(artifacts_path, lm_name)
    tokenizer = AutoTokenizer.from_pretrained(save_path, lowercase=lowercase, use_fast=True, model_max_length=max_length,
                                              config=AutoConfig.from_pretrained(os.path.join(save_path, 'config.json')))
    return tokenizer

def tokenize_text(tokenizer, text, lowercase=False):
    if lowercase:
        text = text.lower()
    
    encoded = tokenizer(text, return_offsets_mapping=True,
                        return_token_type_ids=True, verbose=False)
    
    token_offsets = encoded['offset_mapping'][1:-1]
    tokens = get_tokens_from_offsets(text, token_offsets)

    return tokens, token_offsets

def get_offsets_from_tokens(text, tokens):
    offsets = []
    last_span = 0
    for token in tokens:
        span_start = text.find(token, last_span)
        if span_start == -1:
            raise Exception('Token "%s" (start: %d) not found in text: %s' % (token, last_span, text))
        span_end = span_start + len(token)
        last_span = span_start
        offsets.append((span_start, span_end))
    return offsets

def get_tokens_from_offsets(text, offsets):
    tokens = []
    for (span_start, span_end) in offsets:
        tokens.append(text[span_start:span_end])
    return tokens
