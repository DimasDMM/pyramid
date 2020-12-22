import torch
import torch.nn as nn
from . import *

class CharEncoder(nn.Module):
    def __init__(self, char2id, dimension=60, hidden_size=100, device=None):
        super().__init__()
        self.device = device
        
        self.embedding = create_emb_layer(True, shape=(len(char2id), dimension), device=device)
        init_embeddings(self.embedding.weight)
        
        self.lstm = nn.LSTM(input_size=dimension, hidden_size=hidden_size,
                            bidirectional=True, batch_first=True).to(device=device)
        init_lstm(self.lstm)
        
    def forward(self, inputs):
        x = self.embedding(inputs)
        
        outputs = []
        for seq in x:
            _, (h, _) = self.lstm(seq)
            x_hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
            outputs.append(x_hidden)

        return torch.stack(outputs).to(device=self.device)

class LMEncoder(nn.Module):
    def __init__(self, lm_name, device=None):
        super().__init__()
        self.device = device
        self.lm_layer = create_lm_layer(lm_name, False, device=device)
    
    def _create_lm_layer(self, lm_name, trainable):
        bert_model = BertModel.from_pretrained(lm_name)
        if not trainable:
            for param in bert_model.parameters():
                param.requires_grad = False
        
        return bert_model.to(device=self.device)

    def forward(self, inputs, attention, type_ids, lm_spans, masks):
        x_lm = self.lm_layer(input_ids=inputs, attention_mask=attention, token_type_ids=type_ids,
                             output_hidden_states=True)
        x_lm = torch.stack(x_lm[2][-4:])
        x_lm = torch.mean(x_lm, dim=0)

        x = torch.zeros(size=x_lm.size(), device=self.device)
        for seq_i, seq_span in enumerate(lm_spans):
            mask_length = masks[seq_i].sum()
            
            for token_i, span in enumerate(seq_span):
                if token_i >= mask_length - 1:
                    # Skips from SEP token
                    break
                elif token_i == 0:
                    # Skips CLS token
                    continue
                
                token_i -= 1
                for k in range(span):
                    x[seq_i, token_i] = x[seq_i, token_i].add(x_lm[seq_i, token_i+k+1])
                x[seq_i, token_i].div(span)
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, lm_name, word_embeddings, char2id, char_dimension=60, word_dimension=200,
                 lm_dimension=1024, hidden_size=100, drop_rate=0.45, device=None):
        super(EncoderLayer, self).__init__()
        
        self.device = device

        self.char_encoder = CharEncoder(char2id, char_dimension, hidden_size, device=device)
        self.emb_word = self._create_emb_layer(False, embedding_matrix=word_embeddings, device=device).to(device=device)
        self.dropout = nn.Dropout(drop_rate).to(device=device)
        self.lm_encoder = LMEncoder(lm_name, device=device)

        self.lstm_char = nn.LSTM(input_size=char_dimension, hidden_size=hidden_size,
                                 bidirectional=True, batch_first=True).to(device=device)
        init_lstm(self.lstm_char)
        
        self.lstm_enc = nn.LSTM(input_size=(hidden_size*2 + word_dimension),
                                hidden_size=hidden_size, bidirectional=True, batch_first=True).to(device=device)
        init_lstm(self.lstm_enc)
        
        self.linear = nn.Linear(lm_dimension + hidden_size*2, hidden_size*2).to(device=device)
        init_linear(self.linear)

    def forward(self, input_word, input_char, input_lm, input_lm_attention, input_lm_type_ids,
                input_lm_spans, input_masks):
        x_char = self.char_encoder(input_char)
        x_word = self.emb_word(input_word)
        
        x_enc = torch.cat((x_char, x_word), dim=-1)
        x_enc = self.dropout(x_enc)
        x_enc, _ = self.lstm_enc(x_enc)
        
        x_lm = self.lm_encoder(input_lm, input_lm_attention, input_lm_type_ids, input_lm_spans, input_masks)
        
        x = torch.cat((x_enc, x_lm), dim=2)
        x = self.linear(x)
        
        return x
    
    def _create_emb_layer(self, trainable, embedding_matrix=None, shape=None, device=None):
        if embedding_matrix is None:
            emb_layer = nn.Embedding(num_embeddings=shape[0], embedding_dim=shape[1])
            emb_layer.weight.requires_grad = trainable
        else:
            embedding_tensor = torch.FloatTensor(embedding_matrix).to(device=self.device)
            emb_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=(not trainable))
        return emb_layer
