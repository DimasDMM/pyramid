from collections import defaultdict
import gc
import json
import torch

def evaluate(net, dataloader, device, entity_idx, total_layers=16):
    seq_labels = []
    seq_preds = []

    for batch_data in dataloader:
        # Get inputs
        masks = batch_data['masks'].to(device=device)
        x_word = batch_data['x_word'].to(device=device)
        x_char = batch_data['x_char'].to(device=device)
        x_lm_inputs = batch_data['x_lm_input'].to(device=device)
        x_lm_attention = batch_data['x_lm_attention'].to(device=device)
        x_lm_type_ids = batch_data['x_lm_type_ids'].to(device=device)
        x_lm_span = batch_data['x_lm_spans'].to(device=device)

        y_all_preds = net(x_word, x_char, x_lm_inputs, x_lm_attention, x_lm_type_ids, x_lm_span, masks)
        
        for i_layer in range(total_layers):
            y_targets = batch_data['y_target_%d' % i_layer]
            y_preds = y_all_preds[i_layer].cpu().detach()
            
            for (y_target, y_pred, mask) in zip(y_targets, y_preds, masks):
                mask_cut = int(mask[i_layer:].sum().cpu().detach().numpy())
                y_target = y_target.view(-1)[:mask_cut]
                y_pred = torch.argmax(y_pred, dim=-1).view(-1)[:mask_cut]
                
                seq_labels.append(y_target.cpu().detach().numpy())
                seq_preds.append(y_pred.cpu().detach().numpy())
        
        # Clear some memory
        if device == 'cuda':
            del masks
            del x_word
            del x_char
            del x_lm_inputs
            del x_lm_attention
            del x_lm_type_ids
            del x_lm_span
            gc.collect()
            torch.cuda.empty_cache()

    overall_scores = get_seq_metrics(seq_labels, seq_preds, entity_idx, verbose=1)
    
    return overall_scores

def seq2span(seq, return_types=False, entity_idx=None):
    if entity_idx is not None:
        seq = [entity_idx[x] for x in seq]

    spans = []
    types = []
    _span = _type = None
    for i, t in enumerate(seq):
        if (t[0] == 'B' or t == 'O') and _span is not None:
            spans.append(_span)
            types.append(_type)
            _span = _type = None
        if t[0] == 'B':
            _span = [i, i+1]
            _type = t[2:]
        if t[0] == 'I':
            if _span is not None:
                _span[1] = i+1

    if _span is not None:
        spans.append(_span)
        types.append(_type)
        
    if return_types:
        return spans, types

    return spans

def get_seq_metrics(labels, preds, entity_idx, verbose=0):
    n_correct = n_recall = n_precision = 0
    confusion_dict = defaultdict(lambda: [0, 0, 0]) # n_correct, n_preds, n_labels

    for i in range(len(labels)):
        if verbose > 0:
            print('Evaluating %d out of %d' % (i+1, len(labels)), end='\r')
        
        i_label = labels[i]
        i_pred = preds[i][:len(i_label)]

        spans, types = seq2span(i_pred, True, entity_idx)
        pred_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

        spans, types = seq2span(i_label, True, entity_idx)
        label_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

        correct_set = pred_set & label_set
        
        for _type, _, _ in correct_set:
            confusion_dict[_type][0] += 1
        for _type, _, _ in pred_set:
            confusion_dict[_type][1] += 1
        for _type, _, _ in label_set:
            confusion_dict[_type][2] += 1

        n_correct += len(correct_set)
        n_recall += len(label_set)
        n_precision += len(pred_set)
    
    try:
        recall = n_correct / n_recall
        precision = n_correct / n_precision
        f1 = 2 / (1/recall + 1/precision)
    except:
        recall = precision = f1 = 0

    if verbose > 0:
        print()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_dict': confusion_dict,
    }

def evaluate_save(net, nne_dataset, device, entity_idx, total_layers=16, output_filepath='./artifacts/predictions.json'):
    """
    Evaluate and store results in JSON file.
    """
    global_item_ids = []
    global_texts = []
    global_token_offsets = []
    global_pred_set = []
    global_true_set = []
    n_items = len(nne_dataset)

    for i in range(n_items):
        item_data = nne_dataset.get_item(i)
        item_id = nne_dataset.get_item_id(i)
        text = nne_dataset.get_texts(i)
        token_offsets = nne_dataset.get_token_offsets(i)

        global_item_ids.append(item_id)
        global_texts.append(text)
        global_token_offsets.append(token_offsets)

        # Prepare input data
        masks = torch.unsqueeze(item_data['masks'], 0).to(device=device)
        x_word = torch.unsqueeze(item_data['x_word'], 0).to(device=device)
        x_char = torch.unsqueeze(item_data['x_char'], 0).to(device=device)
        x_lm_inputs = torch.unsqueeze(item_data['x_lm_input'], 0).to(device=device)
        x_lm_attention = torch.unsqueeze(item_data['x_lm_attention'], 0).to(device=device)
        x_lm_type_ids = torch.unsqueeze(item_data['x_lm_type_ids'], 0).to(device=device)
        x_lm_span = torch.unsqueeze(item_data['x_lm_spans'], 0).to(device=device)

        # Make predictions
        y_all_preds = net(x_word, x_char, x_lm_inputs, x_lm_attention, x_lm_type_ids, x_lm_span, masks)
        
        seq_preds = []
        seq_true = []
        for i_layer in range(total_layers):
            y_targets = torch.unsqueeze(item_data['y_target_%d' % i_layer], 0).to(device=device)
            y_preds = y_all_preds[i_layer].cpu().detach()
            
            for (y_target, y_pred, mask) in zip(y_targets, y_preds, masks):
                mask_cut = int(mask[i_layer:].sum().cpu().detach().numpy())
                y_target = y_target.view(-1)[:mask_cut]
                y_pred = torch.argmax(y_pred, dim=-1).view(-1)[:mask_cut]
                
                seq_true.append(y_target.cpu().detach().numpy())
                seq_preds.append(y_pred.cpu().detach().numpy())
        
        for i in range(len(seq_preds)):
            i_true = seq_true[i]
            i_pred = seq_preds[i]

            spans, types = seq2span(i_pred, True, entity_idx)
            pred_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

            spans, types = seq2span(i_true, True, entity_idx)
            true_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

            global_pred_set.append(pred_set)
            global_true_set.append(true_set)
            
        # Clear some memory
        if device == 'cuda':
            del masks
            del x_word
            del x_char
            del x_lm_inputs
            del x_lm_attention
            del x_lm_type_ids
            del x_lm_span
            gc.collect()
            torch.cuda.empty_cache()
    
    save_predictions(global_item_ids, global_texts, global_token_offsets, global_pred_set, global_true_set,
                     total_layers=total_layers, output_filepath=output_filepath)

def predict_save(net, nne_dataset, device, entity_idx, total_layers=16, output_filepath='./artifacts/predictions.json'):
    """
    Evaluate and store results in JSON file.
    """
    global_item_ids = []
    global_texts = []
    global_token_offsets = []
    global_pred_set = []
    n_items = len(nne_dataset)

    for i in range(n_items):
        item_data = nne_dataset.get_item(i)
        item_id = nne_dataset.get_item_id(i)
        text = nne_dataset.get_texts(i)
        token_offsets = nne_dataset.get_token_offsets(i)

        global_item_ids.append(item_id)
        global_texts.append(text)
        global_token_offsets.append(token_offsets)

        # Prepare input data
        masks = torch.unsqueeze(item_data['masks'], 0).to(device=device)
        x_word = torch.unsqueeze(item_data['x_word'], 0).to(device=device)
        x_char = torch.unsqueeze(item_data['x_char'], 0).to(device=device)
        x_lm_inputs = torch.unsqueeze(item_data['x_lm_input'], 0).to(device=device)
        x_lm_attention = torch.unsqueeze(item_data['x_lm_attention'], 0).to(device=device)
        x_lm_type_ids = torch.unsqueeze(item_data['x_lm_type_ids'], 0).to(device=device)
        x_lm_span = torch.unsqueeze(item_data['x_lm_spans'], 0).to(device=device)

        # Make predictions
        y_all_preds = net(x_word, x_char, x_lm_inputs, x_lm_attention, x_lm_type_ids, x_lm_span, masks)
        
        seq_preds = []
        for i_layer in range(total_layers):
            y_preds = y_all_preds[i_layer].cpu().detach()
            
            for (y_pred, mask) in zip(y_preds, masks):
                mask_cut = int(mask[i_layer:].sum().cpu().detach().numpy())
                y_pred = torch.argmax(y_pred, dim=-1).view(-1)[:mask_cut]
                
                seq_preds.append(y_pred.cpu().detach().numpy())
        
        for i in range(len(seq_preds)):
            i_pred = seq_preds[i]
            spans, types = seq2span(i_pred, True, entity_idx)
            pred_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}
            global_pred_set.append(pred_set)
            
        # Clear some memory
        if device == 'cuda':
            del masks
            del x_word
            del x_char
            del x_lm_inputs
            del x_lm_attention
            del x_lm_type_ids
            del x_lm_span
            gc.collect()
            torch.cuda.empty_cache()
    
    save_predictions(global_item_ids, global_texts, global_token_offsets, global_pred_set, targets=None,
                     total_layers=total_layers, output_filepath=output_filepath)

def save_predictions(dataset_ids, dataset_texts, dataset_token_offsets, predictions, targets=None,
                     total_layers=16, output_filepath='./artifacts/predictions.json'):
    results = []
    n_items = len(dataset_texts)

    for i in range(n_items):
        item_text = dataset_texts[i]
        item_token_offsets = dataset_token_offsets[i]
        item_id = dataset_ids[i]
        
        pred_entities = []
        item_predictions = predictions[(i*total_layers):((i+1)*total_layers)]
        for i_layer, layer_predictions in enumerate(item_predictions):
            for layer_prediction in layer_predictions:
                layer_prediction = list(layer_prediction)
                entity_type = layer_prediction[0]
                token_start = layer_prediction[1]
                token_end = layer_prediction[2] + i_layer - 1

                span_start = item_token_offsets[token_start][0]
                span_end = item_token_offsets[token_end][1]

                pred_entities.append({
                    'entity_type': entity_type,
                    'span': [span_start, span_end],
                    'text': item_text[span_start:span_end],
                })
        
        results.append({
            'item_id': item_id,
            'text': item_text,
            'entities': pred_entities,
        })

    with open(output_filepath, 'w') as outfile:
        json.dump(results, outfile)
