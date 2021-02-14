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

def run_predictor(logger, config: Config):
    logger.info(config.__dict__)

    # Set up step
    logger.info('== PREDICTOR ==')

    if config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        torch.cuda.set_device(config.device)
        config.device = device

    logger.info('Using device: %s' % config.device)

    # Load datasets
    logger.info('Loading dataset...')

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
    test_dataset, _ = add_layer_outputs(test_dataset, total_layers=model_params.total_layers, entity_dict=entity_dict)

    # Create tokenizer and data inputs
    logger.info('Loading tokenizer and data inputs...')
    tokenizer = get_tokenizer(lm_name=model_params.lm_name, lowercase=(not model_params.cased_lm))
    word_input = WordInput(word2id, lowercase=(not model_params.cased_word))
    char_input = CharInput(char2id, lowercase=(not model_params.cased_char))
    lm_input = LMInput(tokenizer, lowercase=(not model_params.cased_lm))

    logger.info('Creating data loader...')
    nne_test_dataset = NestedNamedEntitiesDataset(
            test_dataset, word_input, char_input, lm_input, total_layers=model_params.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512, has_outputs=False)
    
    # Store predictions
    logger.info('Saving predictions as JSON file...')
    logger.info('- Test set...')
    predict_save(net, nne_test_dataset, config.device, entity_idx, total_layers=model_params.total_layers,
                 output_filepath='./artifacts/%s_test_predictions.json' % config.dataset)

    logger.info('Done')
