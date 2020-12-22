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

def run_training(logger, config: Config):
    # Set up step
    logger.info('== SET UP ==')

    best_f1 = 0.
    if config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        torch.cuda.set_device(config.device)

    logger.info('Using device: %s' % config.device)

    # Load datasets
    logger.info('Loading datasets...')

    train_file = './data/train.%s.json' % config.dataset
    with open(train_file, 'r') as fp:
        train_dataset = json.load(fp)
    logger.info('Loaded train dataset size: %d' % len(train_dataset))

    valid_file = './data/valid.%s.json' % config.dataset
    with open(valid_file, 'r') as fp:
        valid_dataset = json.load(fp)
    logger.info('Loaded valid dataset size: %d' % len(valid_dataset))

    # Load trained model or build a new one
    if config.continue_training:
        if not os.path.exists(config.model_ckpt):
            raise Exception('Pretrained model not exists in: %s' % config.model_ckpt)

        net, config, word2id, char2id, entity_idx = load_model_objects(
                logger, config.model_ckpt, config.dataset, config.device)
        entity_dict = {x:i for i, x in enumerate(entity_idx)}
        if hasattr(config, 'f1_score'):
            best_f1 = config.f1_score
        
        # Transform format of nested entities
        logger.info('Building layer outputs...')
        train_dataset, _ = add_layer_outputs(train_dataset, total_layers=config.total_layers, entity_dict=entity_dict)
        valid_dataset, _ = add_layer_outputs(valid_dataset, total_layers=config.total_layers, entity_dict=entity_dict)
    else:
        if os.path.exists(config.model_ckpt):
            raise Exception('Pretrained model already exists in: %s' % config.model_ckpt)

        # Transform format of nested entities
        logger.info('Building layer outputs...')
        train_dataset, entity_dict = add_layer_outputs(train_dataset, total_layers=config.total_layers)
        valid_dataset, _ = add_layer_outputs(valid_dataset, total_layers=config.total_layers, entity_dict=entity_dict)
        entity_idx = [x for x in list(entity_dict.keys())]

        # Load embeddings and build vocabularies
        logger.info('Loading embeddings...')
        special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
        embedding_matrix, id2word, word2id = load_embedding_matrix(config.wv_file, config.token_emb_dim, special_tokens)
        id2char, char2id = build_char_vocab(train_dataset, special_tokens=special_tokens)

        # Build model
        logger.info('Building model...')
        total_classes = len(entity_idx)
        net = PyramidNet(embedding_matrix, char2id, lm_name=config.lm_name, total_layers=config.total_layers,
                         drop_rate=config.dropout, seq_length=512, lm_dimension=config.lm_emb_dim,
                         char_dimension=config.char_emb_dim, word_dimension=config.token_emb_dim,
                         total_classes=total_classes, device=config.device)

    # Create tokenizer and data inputs
    logger.info('Loading tokenizer and data inputs...')
    tokenizer = get_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))
    word_input = WordInput(word2id)
    char_input = CharInput(char2id)
    bert_input = BertInput(tokenizer)

    # Data loaders
    logger.info('Creating data loaders...')
    nne_train_dataset = NestedNamedEntitiesDataset(
            train_dataset, word_input, char_input, bert_input, total_layers=config.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    nne_valid_dataset = NestedNamedEntitiesDataset(
            valid_dataset, word_input, char_input, bert_input, total_layers=config.total_layers,
            skip_exceptions=False, max_items=-1, padding_length=512)
    train_dataloader = DataLoader(nne_train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    valid_dataloader = DataLoader(nne_valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Training step
    logger.info('== MODEL TRAINING ==')
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    gradient_clip = 5.

    history = []
    step = 0
    n_batches = len(train_dataloader)

    logger.info('Start training')

    for i_epoch in range(config.max_epoches):
        run_loss = 0

        # Set model to training mode
        net.train()

        for i_batch, batch_data in enumerate(train_dataloader):
            step += 1

            # Get inputs
            masks = batch_data['masks'].to(device=config.device)
            x_word = batch_data['x_word'].to(device=config.device)
            x_char = batch_data['x_char'].to(device=config.device)
            x_lm_inputs = batch_data['x_lm_input'].to(device=config.device)
            x_lm_attention = batch_data['x_lm_attention'].to(device=config.device)
            x_lm_type_ids = batch_data['x_lm_type_ids'].to(device=config.device)
            x_lm_span = batch_data['x_lm_spans'].to(device=config.device)
            
            y_all_targets = []
            for i_layer in range(config.total_layers):
                y_target = batch_data['y_target_%d' % i_layer].to(device=config.device)
                y_all_targets.append(y_target)

            # Predict entities
            y_all_preds = net(x_word, x_char, x_lm_inputs, x_lm_attention, x_lm_type_ids, x_lm_span, masks)

            # Compute loss
            loss = 0
            for i_pred, y_pred_logits in enumerate(y_all_preds):
                loss_tensor = criterion(y_pred_logits.permute(0, -1, 1), y_all_targets[i_pred])
                loss += (loss_tensor * masks[:,i_pred:]).sum()

            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(net.parameters(), gradient_clip) # Avoid gradient exploding issue
            optimizer.step()
            
            adjust_lr(optimizer, step)
            run_loss += loss.cpu().data.numpy()

            if i_batch % 10 == 0:
                logger.info("Epoch %d of %d | Batch %d of %d | Loss = %.3f" % (
                        i_epoch + 1, config.max_epoches, i_batch + 1, n_batches, run_loss / (i_batch + 1)))
            
            if config.max_steps != -1 and config.max_steps <= step:
                break
            
            # Clear some memory
            if config.device == 'cuda':
                del masks
                del x_word
                del x_char
                del x_lm_inputs
                del x_lm_attention
                del x_lm_type_ids
                del x_lm_span
                gc.collect()
                torch.cuda.empty_cache()

        history.append(run_loss / len(train_dataloader))
        logger.info("Epoch %d of %d | Loss = %.3f" % (i_epoch + 1, config.max_epoches,
                                                      run_loss / len(train_dataloader)))

        logger.info('Evaluating model with val. dataset...')
        net.eval() # Set model to eval mode
        eval_scores = evaluate(
                net, dataloader=valid_dataloader, device=config.device,
                total_layers=config.total_layers, entity_idx=entity_idx)
        logger.info('Val. Scores | Precision: %.4f | Recall: %.4f | F1-score: %.4f | Best: %.4f' % (
                eval_scores['precision'], eval_scores['recall'], eval_scores['f1'], best_f1))
        
        if eval_scores['f1'] > best_f1:
            logger.info('F1 Score Improved! Saving model...')
            best_f1 = eval_scores['f1']
            config.f1_score = best_f1
            save_model_objects(net, config, word2id, char2id, entity_idx)

        if config.max_steps != -1 and config.max_steps <= step:
            break
    
    logger.info('End training')

def adjust_lr(optimizer, step, decay_rate=0.05, decay_steps=1000, inital_lr=0.01):
    """
    Ajusts Learnin-Rate using the formula described in the paper
    """
    lr = inital_lr / (1 + decay_rate * step / decay_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
