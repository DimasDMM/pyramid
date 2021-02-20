import torch
from torch.utils.data import Dataset
from .tokenization import *

class NestedNamedEntitiesDataset(Dataset):    
    def __init__(self, data, lm_input, word_input=None, char_input=None, padding_length=512,
                 total_layers=16, skip_exceptions=True, max_items=-1, has_outputs=True):
        self.total_layers = total_layers
        self.has_outputs = has_outputs
        
        self.item_id = []
        self.texts = []
        self.token_offsets = []
        self.masks = []
        self.X_word = []
        self.X_char = []
        self.X_lm_inputs = []
        self.X_lm_attention = []
        self.X_lm_type_ids = []
        self.X_lm_spans = []
        self.Y_entities = [[] for _ in range(total_layers)]
        self.n_skipped = 0
        
        for i, item in enumerate(data):
            if max_items > 0 and i >= max_items:
                break
            
            try:
                x_word = word_input.encode(item['tokens'], padding_length) if word_input is not None else []
                x_char = char_input.encode(item['tokens'], padding_length) if char_input is not None else []
                x_lm, x_lm_span = lm_input.encode(item['tokens'], padding_length)

                mask = [1.] * len(item['tokens']) + [0.] * (padding_length - len(item['tokens']))
                self.masks.append(mask)
                
                token_offsets = get_offsets_from_tokens(item['text'], item['tokens'])
                self.token_offsets.append(token_offsets)

                if 'item_id' in item:
                    self.item_id.append(item['item_id'])
                else:
                    self.item_id.append(i)

                self.texts.append(item['text'])
                self.X_word.append(x_word)
                self.X_char.append(x_char)

                self.X_lm_inputs.append(x_lm[0])
                self.X_lm_attention.append(x_lm[1])
                self.X_lm_type_ids.append(x_lm[2])
                self.X_lm_spans.append(x_lm_span)
                
                if has_outputs:
                    for i_layer, raw_seq in enumerate(item['layer_outputs']):
                        if i_layer >= total_layers:
                            break
                        padded_seq = raw_seq if len(raw_seq) > 0 else []
                        if len(padded_seq) < padding_length - i_layer:
                            padded_seq = padded_seq + [0] * (padding_length - len(padded_seq) - i_layer)
                        self.Y_entities[i_layer].append(padded_seq)
            except Exception as e:
                # Text-too-long exception
                if skip_exceptions:
                    self.n_skipped += 1
                    continue
                else:
                    raise e
        
        self._init_tensors(padding_length, total_layers)

    def _init_tensors(self, padding_length, total_layers):
        self.masks = torch.tensor(self.masks, dtype=torch.float)
        self.X_word = torch.tensor(self.X_word, dtype=torch.long)
        self.X_char = torch.tensor(self.X_char, dtype=torch.long)
        self.X_lm_inputs = torch.tensor(self.X_lm_inputs, dtype=torch.long)
        self.X_lm_attention = torch.tensor(self.X_lm_attention, dtype=torch.long)
        self.X_lm_type_ids = torch.tensor(self.X_lm_type_ids, dtype=torch.long)
        self.X_lm_spans = torch.tensor(self.X_lm_spans, dtype=torch.long)
        if self.has_outputs:
            for i, seqs in enumerate(self.Y_entities):
                self.Y_entities[i] = torch.tensor(seqs, dtype=torch.long)

    def __len__(self):
        return len(self.X_word)

    def __getitem__(self, idx):
        data = {
            'masks': self.masks[idx],
            'x_word': self.X_word[idx],
            'x_char': self.X_char[idx],
            'x_lm_input': self.X_lm_inputs[idx],
            'x_lm_attention': self.X_lm_attention[idx],
            'x_lm_type_ids': self.X_lm_type_ids[idx],
            'x_lm_spans': self.X_lm_spans[idx],
        }
        if self.has_outputs:
            for i_layer in range(self.total_layers):
                data['y_target_%d' % i_layer] = self.Y_entities[i_layer][idx]
        return data

    def get_item(self, idx):
        return self.__getitem__(idx)
    
    def get_item_id(self, idx):
        return self.item_id[idx]
    
    def get_texts(self, idx):
        return self.texts[idx]
    
    def get_token_offsets(self, idx):
        return self.token_offsets[idx]

    def get_n_skipped(self):
        return self.n_skipped
