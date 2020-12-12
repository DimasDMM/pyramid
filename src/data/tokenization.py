import os
import re
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def get_tokenizer(artifacts_path='artifacts/', lm_name='dmis-lab/biobert-v1.1', lowercase=True):
    slow_tokenizer = BertTokenizer.from_pretrained(lm_name)

    save_path = '%s%s/' % (artifacts_path, lm_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        slow_tokenizer.save_pretrained(save_path)

    # We can already use the Slow Tokenizer, but its implementation in Rust is much faster.
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
