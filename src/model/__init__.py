import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

def create_emb_layer(trainable, embedding_matrix=None, shape=None, device=None):
    if embedding_matrix is None:
        emb_layer = nn.Embedding(num_embeddings=shape[0], embedding_dim=shape[1])
        emb_layer.weight.requires_grad = trainable
    else:
        embedding_tensor = torch.FloatTensor(embedding_matrix)
        emb_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=(not trainable))
    return emb_layer.to(device=device)
    
def create_lm_layer(lm_name, trainable, device=None, artifacts_path='./artifacts/'):
    lm_path = '%s/%s/' % (artifacts_path, lm_name)
    pretrained_lm = AutoModel.from_pretrained(lm_path)
    if not trainable:
        for param in pretrained_lm.parameters():
            param.requires_grad = False
    return pretrained_lm.to(device=device)
