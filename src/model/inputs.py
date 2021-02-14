
class WordInput:
    def __init__(self, word2id, lowercase=True):
        self.word2id = word2id
        self.lowercase = lowercase
        
    def encode(self, tok_text, padding_length=512, unk='<unk>', pad='<pad>'):
        if self.lowercase:
            tok_text = [w.lower() for w in tok_text]

        input_ids = [self.word2id[word] if word in self.word2id else self.word2id[unk] for word in tok_text]

        # Pad if necessary
        add_padding = padding_length - len(input_ids)
        if add_padding > 0:
            pad_id = self.word2id[pad]
            input_ids = input_ids + ([pad_id] * add_padding)
        elif add_padding < 0:
            raise Exception('(Words) Text too long (%d / %d):' % (len(input_ids), padding_length), tok_text)

        return input_ids

class CharInput:
    def __init__(self, char2id, lowercase=True):
        self.char2id = char2id
        self.lowercase = lowercase

    def encode(self, tok_text, padding_length=512, char_padding=128, unk='<unk>', pad='<pad>'):
        if self.lowercase:
            tok_text = [w.lower() for w in tok_text]

        input_ids = []
        
        for token in tok_text:
            char_ids = [self.char2id[char] if char in self.char2id else self.char2id[unk] for char in token]

            # Pad char list if necessary
            add_padding = char_padding - len(char_ids)
            if add_padding > 0:
                pad_id = self.char2id[pad]
                char_ids = char_ids + ([pad_id] * add_padding)
            elif add_padding < 0:
                raise Exception('(Chars-1) Text too long (%d / %d):' % (len(char_ids), char_padding), tok_text)

            input_ids.append(char_ids)

        # Pad token list if necessary
        add_padding = padding_length - len(input_ids)
        if add_padding > 0:
            pad_id = self.char2id[pad]
            input_ids = input_ids + ([[pad_id] * char_padding] * add_padding)
        elif add_padding < 0:
            raise Exception('(Chars-2) Text too long (%d / %d):' % (len(input_ids), padding_length), tok_text)
        
        return input_ids

class LMInput:
    def __init__(self, tokenizer, lowercase=True):
        self.tokenizer = tokenizer
        self.lowercase = lowercase

    def encode(self, tok_text, padding_length=512):
        if self.lowercase:
            tok_text = [w.lower() for w in tok_text]

        tok_text = ['[CLS]'] + tok_text + ['[SEP]']
        pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        # Encode context (token IDs, mask and token types)
        token_spans = [1]
        input_ids = []
        type_ids = []
        attention_mask = []
        for token in tok_text:
            encoded_text = self.tokenizer.encode(token)

            # Create inputs
            span = len(encoded_text['input_ids'][1:-1])
            token_spans += [span] + [0] * (span - 1)
            input_ids += encoded_text['input_ids'][1:-1]
            type_ids += encoded_text['token_type_ids'][1:-1]
            attention_mask += encoded_text['attention_mask'][1:-1]
        
        if len(input_ids) > padding_length:
            raise Exception('(LM) Text too long (%d / %d): %s' % (len(input_ids), padding_length, tok_text))

        # Pad if necessary. Note that "pad_idx" is the ID of the pad token in LM.
        add_padding = padding_length - len(input_ids)
        if add_padding > 0:
            input_ids = input_ids + ([pad_idx] * add_padding)
            attention_mask = attention_mask + ([0] * add_padding)
            type_ids = type_ids + ([0] * add_padding)
        
        token_spans += ([1] * (padding_length - len(token_spans)))

        # LM inputs must be as follows: input_ids, attention_mask, token_type_ids
        return [input_ids, attention_mask, type_ids], token_spans
