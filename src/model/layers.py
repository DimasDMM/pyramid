import torch
import torch.nn as nn
from .encoders import *

class DecodingLayer(nn.Module):
    def __init__(self, drop_rate=0.4, seq_length=512, hidden_size=100, device=None):
        super(DecodingLayer, self).__init__()
        self.device = device
        
        self.norm = nn.LayerNorm(normalized_shape=(seq_length, hidden_size*2)).to(device=device)
        self.dropout_1 = nn.Dropout(drop_rate).to(device=device)
        self.dropout_2 = nn.Dropout(drop_rate).to(device=device)
        
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, bidirectional=True, batch_first=True).to(device=device)
        self.conv = nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=2, stride=1).to(device=device)
    
    def forward(self, input):
        x = self.norm(input)
        x = self.dropout_1(x)
        
        x, _ = self.lstm(x)
        x = self.dropout_2(x)
        
        h = x
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(2, 1)
        
        return h, x

class PyramidLayer(nn.Module):
    def __init__(self, total_layers=16, drop_rate=0.4, seq_length=512, hidden_size=100, device=None):
        super(PyramidLayer, self).__init__()
        self.device = device
        self.total_layers = total_layers
        self.seq_length = seq_length
        self.decoding_layers = nn.ModuleList([DecodingLayer(drop_rate=drop_rate, seq_length=seq_length-i,
                                                            hidden_size=hidden_size, device=device) for i in range(total_layers)])
    
    def forward(self, input):
        h = []
        x_layer = input
        for i, layer in enumerate(self.decoding_layers):
            h_layer, x_layer = layer(x_layer)
            h.append(h_layer)
        
        return h

class InverseDecodingLayer(nn.Module):
    def __init__(self, drop_rate=0.4, seq_length=512, hidden_size=100, device=None):
        super(InverseDecodingLayer, self).__init__()
        self.device = device
        
        self.norm = nn.LayerNorm(normalized_shape=(seq_length, hidden_size*2)).to(device=device)
        self.dropout_1 = nn.Dropout(drop_rate).to(device=device)
        self.dropout_2 = nn.Dropout(drop_rate).to(device=device)
        
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, bidirectional=True, batch_first=True).to(device=device)
        self.conv = nn.Conv1d(in_channels=hidden_size*4, out_channels=hidden_size*2, kernel_size=2, padding=1, stride=1).to(device=device)
    
    def forward(self, input_h, input_x):
        x = self.norm(input_x)
        x = self.dropout_1(x)
        
        x, _ = self.lstm(x)
        x = self.dropout_2(x)
        
        x = torch.cat((input_h, x), dim=2)
        
        h = x
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(2, 1)
        
        return h, x

class InversePyramidLayer(nn.Module):
    def __init__(self, total_layers=16, drop_rate=0.4, seq_length=512, hidden_size=100, device=None):
        super(InversePyramidLayer, self).__init__()
        self.device = device
        self.total_layers = total_layers
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.idecoding_layers = nn.ModuleList([InverseDecodingLayer(drop_rate=drop_rate, seq_length=seq_length-i,
                                                                    hidden_size=hidden_size, device=device) for i in range(total_layers)])
    
    def forward(self, input_hs):
        h = []
        batch_size = input_hs[-1].size()[0]
        
        x_pad = torch.zeros(batch_size, 1, self.hidden_size*4).to(device=self.device)
        x_layer = torch.zeros(batch_size,
                              self.seq_length - self.total_layers + 1,
                              self.hidden_size*2).to(device=self.device)
        
        for i, layer in reverse_enumerate(self.idecoding_layers):
            h_layer, x_layer = layer(input_hs[i], x_layer)
            #x_layer = torch.cat((x_pad, x_layer), dim=1)
            h.append(h_layer)
        
        h.reverse()
        return h

class PyramidNet(nn.Module):
    def __init__(self, embedding_matrix, char_vocab, lm_name='dmis-lab/biobert-v1.1', total_layers=16,
                 drop_rate=0.4, seq_length=512, char_dimension=100, word_dimension=100,
                 lm_dimension=768, hidden_size=100, total_classes=10, device=None):
        super(PyramidNet, self).__init__()
        self.device = device
        
        self.encoder_layer = EncoderLayer(
                lm_name, embedding_matrix, char_vocab, char_dimension=char_dimension,
                word_dimension=word_dimension, lm_dimension=lm_dimension, hidden_size=hidden_size,
                drop_rate=drop_rate, device=device)
        
        self.pyramid = PyramidLayer(total_layers=total_layers, drop_rate=drop_rate,
                                    seq_length=seq_length, hidden_size=hidden_size, device=device)
        self.inverse_pyramid = InversePyramidLayer(total_layers=total_layers, drop_rate=drop_rate,
                                                   seq_length=seq_length, hidden_size=hidden_size, device=device)
        self.linear = nn.Linear(hidden_size*4, total_classes).to(device=self.device)
    
    def forward(self, input_word, input_char, input_lm, input_lm_attention,
                input_lm_type_ids, input_lm_spans, input_masks):
        x = self.encoder_layer(input_word, input_char, input_lm, input_lm_attention,
                               input_lm_type_ids, input_lm_spans, input_masks)
        x = self.pyramid(x)
        x = self.inverse_pyramid(x)
        x = [self.linear(x_layer) for x_layer in x]
        return x

def reverse_enumerate(L):
    i = len(L)
    while i > 0:
        i -= 1
        yield i, L[i]
