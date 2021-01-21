import json
import os
import pickle
import torch

from ..data.tokenization import *
from ..data.embeddings import *
from .inputs import *
from .layers import *

def load_model_objects(logger, model_ckpt, dataset, device):
    logger.info('Loading config params...')
    filepath = '%s%s_config.pickle' % (model_ckpt, dataset)
    with open(filepath, 'rb') as fp:
        model_config = pickle.load(fp)
        model_params = model_config['config']
        word2id = model_config['word2id']
        char2id = model_config['char2id']
        entity_idx = model_config['entity_idx']

    # Load datasets
    logger.info('Loading test dataset...')

    test_file = './data/test.%s.json' % dataset
    with open(test_file, 'r') as fp:
        test_dataset = json.load(fp)
    logger.info('Loaded test dataset size: %d' % len(test_dataset))

    # Load embeddings and build vocabularies
    logger.info('Loading embeddings...')
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]', '<unk>', '<pad>']
    embedding_matrix, _, _ = load_embedding_matrix(model_params.wv_file, model_params.token_emb_dim, special_tokens)

    # Build base model without trained weights
    logger.info('Building base model...')
    total_classes = len(entity_idx)
    net = PyramidNet(embedding_matrix, char2id, lm_name=model_params.lm_name, total_layers=model_params.total_layers,
                     drop_rate=model_params.dropout, seq_length=512, lm_dimension=model_params.lm_emb_dim,
                     char_dimension=model_params.char_emb_dim, word_dimension=model_params.token_emb_dim,
                     total_classes=total_classes, device=device)

    # Load model weights and config params
    logger.info('Loading model weights...')
    filepath = '%s%s_model.pt' % (model_ckpt, dataset)
    net.load_state_dict(torch.load(filepath))
    net.eval()

    return net, model_params, word2id, char2id, entity_idx

def save_model_objects(net, config, word2id, char2id, entity_idx):
    if not os.path.exists(config.model_ckpt):
        os.makedirs(config.model_ckpt)

    filepath = '%s%s_model.pt' % (config.model_ckpt, config.dataset)
    torch.save(net.state_dict(), filepath)

    model_config = {
        'config': config,
        'word2id': word2id,
        'char2id': char2id,
        'entity_idx': entity_idx,
    }
    filepath = '%s%s_config.pickle' % (config.model_ckpt, config.dataset)
    with open(filepath, 'wb') as fp:
        pickle.dump(model_config, fp, protocol=pickle.HIGHEST_PROTOCOL)
