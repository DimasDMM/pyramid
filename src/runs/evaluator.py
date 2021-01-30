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
    logger.info(config.__dict__)

    # Set up step
    logger.info('== EVALUATOR ==')

    if config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        torch.cuda.set_device(config.device)
        config.device = device

    logger.info('Using device: %s' % config.device)

    # Load datasets
    logger.info('Loading dataset...')

    valid_file = './data/valid.%s.json' % config.dataset
    with open(valid_file, 'r') as fp:
        valid_dataset = json.load(fp)
    logger.info('Loaded valid dataset size: %d' % len(valid_dataset))

    test_file = './data/test.%s.json' % config.dataset
    with open(test_file, 'r') as fp:
        test_dataset = json.load(fp)
    logger.info('Loaded test dataset size: %d' % len(test_dataset))

    # Load trained model
    logger.info('Loading model...')
    net, model_params, word2id, char2id, entity_idx = load_model_objects(
            logger, config.model_ckpt, config.dataset, config.device)
    entity_dict = {x:i for i, x in enumerate(entity_idx)}
    net.eval() # Set model to eval mode
    
    # Transform format of nested entities
    logger.info('Building layer outputs...')
    valid_dataset, _ = add_layer_outputs(valid_dataset, total_layers=model_params.total_layers, entity_dict=entity_dict)
    test_dataset, _ = add_layer_outputs(test_dataset, total_layers=model_params.total_layers, entity_dict=entity_dict)

    # Create tokenizer and data inputs
    logger.info('Loading tokenizer and data inputs...')
    tokenizer = get_tokenizer(lm_name=model_params.lm_name, lowercase=(not model_params.cased_lm))
    word_input = WordInput(word2id, lowercase=(not model_params.cased_word))
    char_input = CharInput(char2id, lowercase=(not model_params.cased_char))
    bert_input = BertInput(tokenizer, lowercase=(not model_params.cased_lm))

    logger.info('Creating data loader...')
    nne_valid_dataset = NestedNamedEntitiesDataset(
            valid_dataset, word_input, char_input, bert_input, total_layers=model_params.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    valid_dataloader = DataLoader(nne_valid_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)
    
    nne_test_dataset = NestedNamedEntitiesDataset(
            test_dataset, word_input, char_input, bert_input, total_layers=model_params.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    test_dataloader = DataLoader(nne_test_dataset, batch_size=model_params.batch_size, shuffle=False, num_workers=0)
    
    # Evaluate with test dataset
    logger.info('Evaluating model with dev/test datasets...')
    eval_scores = evaluate(net, dataloader=valid_dataloader, device=config.device,
                           total_layers=model_params.total_layers, entity_idx=entity_idx)
    logger.info('Dev Scores | Precision: %.4f | Recall: %.4f | F1-score: %.4f' % (
                eval_scores['precision'], eval_scores['recall'], eval_scores['f1']))

    eval_scores = evaluate(net, dataloader=test_dataloader, device=config.device,
                           total_layers=model_params.total_layers, entity_idx=entity_idx)
    logger.info('Test Scores | Precision: %.4f | Recall: %.4f | F1-score: %.4f' % (
                eval_scores['precision'], eval_scores['recall'], eval_scores['f1']))
    
    # Store predictions
    logger.info('Saving predictions as JSON file...')
    logger.info('- Validation set...')
    evaluate_save(net, nne_valid_dataset, config.device, entity_idx, total_layers=model_params.total_layers,
                  output_filepath='./artifacts/valid_predictions.json')
                  
    logger.info('- Test set...')
    evaluate_save(net, nne_test_dataset, config.device, entity_idx, total_layers=model_params.total_layers,
                  output_filepath='./artifacts/test_predictions.json')

    logger.info('Done')
