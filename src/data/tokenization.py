import os
import re
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def get_tokenizer(artifacts_path='./artifacts/', lm_name='dmis-lab/biobert-large-cased-v1.1', lowercase=True):
    save_path = '%s%s/' % (artifacts_path, lm_name)
    tokenizer = BertWordPieceTokenizer('%svocab.txt' % save_path, lowercase=lowercase)
    return tokenizer

def tokenize_text(tokenizer, text, lowercase=True):
    if lowercase:
        text = text.lower()
    
    encoded = tokenizer.encode(text)
    
    tokens = encoded.tokens[1:-1]
    spans = encoded.offsets[1:-1]
    
    spans = [[x[0], x[1]-1] for x in spans]
    
    i = len(tokens)
    while i >= 0:
        i -= 1
        if re.search(r"^##.+", tokens[i]):
            token = tokens[i][2:]
            tokens[i-1] += token
            spans[i-1][1] += len(token)
            del tokens[i]
            del spans[i]

    return tokens, spans
