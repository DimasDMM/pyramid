import json
import gc
import os
import pickle
import torch
from torch.utils.data import DataLoader

from .. import *
from ..data.dataloader import *
from ..data.embeddings import *
from ..data.preprocess import *
from ..data.tokenization import *
from ..evaluation.overall import *
from ..model.encoders import *
from ..model.inputs import *
from ..model.layers import *
from ..utils.config import *

def run_training(logger, config: Config):
    # Set up step
    logger.info('== EVALUATOR ==')

    if config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        torch.cuda.set_device(config.device)

    logger.info('Using device: %s' % config.device)

    logger.info('Loading config params...')
    filepath = '%s%s_config.pickle' % (config.model_ckpt, config.dataset)
    with open(filepath, 'rb') as fp:
        model_config = pickle.load(fp)
        model_params = model_config['config']
        word2id = model_config['word2id']
        char2id = model_config['char2id']
        entity_idx = model_config['entity_idx']

    # Load datasets
    logger.info('Loading test dataset...')

    test_file = './data/test.%s.json' % config.dataset
    with open(test_file, 'r') as fp:
        test_dataset = json.load(fp)
    logger.info('Loaded test dataset size: %d' % len(test_dataset))

    # Transform format of nested entities
    logger.info('Building layer outputs...')
    entity_dict = {x:i for i, x in enumerate(entity_idx)}
    test_dataset, _ = add_layer_outputs(test_dataset, total_layers=model_params.total_layers, entity_dict=entity_dict)

    # Load embeddings and build vocabularies
    logger.info('Loading embeddings...')
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
    embedding_matrix, _, _ = load_embedding_matrix(model_params.wv_file, model_params.token_emb_dim, special_tokens)

    # Create tokenizer and data inputs
    logger.info('Loading tokenizer and data inputs...')
    tokenizer = get_tokenizer(lm_name=model_params.lm_name, lowercase=(not model_params.cased))
    word_input = WordInput(word2id)
    char_input = CharInput(char2id)
    bert_input = BertInput(tokenizer)

    logger.info('Creating data loader...')
    nne_test_dataset = NestedNamedEntitiesDataset(
            test_dataset, word_input, char_input, bert_input, total_layers=model_params.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    test_dataloader = DataLoader(nne_test_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)

    # Build base model without trained weights
    logger.info('Building base model...')
    total_classes = len(entity_idx)
    net = PyramidNet(embedding_matrix, char2id, lm_name=model_params.lm_name, total_layers=model_params.total_layers,
                     drop_rate=model_params.dropout, seq_length=512, lm_dimension=model_params.lm_emb_dim,
                     total_classes=total_classes, device=config.device)

    # Load model weights and config params
    logger.info('Loading model weights...')
    filepath = '%s%s_model.pt' % (config.model_ckpt, config.dataset)
    net.load_state_dict(torch.load(filepath))
    net.eval()
    
    # Evaluate with test dataset
    logger.info('Evaluating model with test dataset...')
    eval_scores = evaluate(net, dataloader=test_dataloader, device=config.device,
                           total_layers=model_params.total_layers, entity_idx=entity_idx)
    logger.info('Test Scores | Precision: %.4f | Recall: %.4f | F1-score: %.4f' % (
                eval_scores['precision'], eval_scores['recall'], eval_scores['f1']))

    logger.info('Done')
