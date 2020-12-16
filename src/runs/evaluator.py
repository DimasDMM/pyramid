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
from ..model.manager import *
from ..utils.config import *

def run_evaluator(logger, config: Config):
    # Set up step
    logger.info('== EVALUATOR ==')

    if config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        torch.cuda.set_device(config.device)
    logger.info('Using device: %s' % config.device)

    net, model_params, word2id, char2id, entity_idx = load_model_objects(
            logger, config.model_ckpt, config.dataset, config.device)
    
    # Create tokenizer and data inputs
    logger.info('Loading tokenizer and data inputs...')
    tokenizer = get_tokenizer(lm_name=model_params.lm_name, lowercase=(not model_params.cased))
    word_input = WordInput(word2id)
    char_input = CharInput(char2id)
    bert_input = BertInput(tokenizer)

    # Transform format of nested entities
    logger.info('Building layer outputs...')
    entity_dict = {x:i for i, x in enumerate(entity_idx)}
    test_dataset, _ = add_layer_outputs(test_dataset, total_layers=model_params.total_layers, entity_dict=entity_dict)

    logger.info('Creating data loader...')
    nne_test_dataset = NestedNamedEntitiesDataset(
            test_dataset, word_input, char_input, bert_input, total_layers=model_params.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    test_dataloader = DataLoader(nne_test_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)
    
    # Evaluate with test dataset
    logger.info('Evaluating model with test dataset...')
    eval_scores = evaluate(net, dataloader=test_dataloader, device=config.device,
                           total_layers=model_params.total_layers, entity_idx=entity_idx)
    logger.info('Test Scores | Precision: %.4f | Recall: %.4f | F1-score: %.4f' % (
                eval_scores['precision'], eval_scores['recall'], eval_scores['f1']))

    logger.info('Done')
