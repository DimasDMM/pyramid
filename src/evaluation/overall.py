import torch
from collections import defaultdict

def evaluate(net, dataloader, device, entity_idx, total_layers=16):
    seq_labels = []
    seq_preds = []

    for i_batch, batch_data in enumerate(dataloader):
        # Get inputs
        masks = batch_data['masks'].to(device=device)
        x_word = batch_data['x_word'].to(device=device)
        x_char = batch_data['x_char'].to(device=device)
        x_lm_inputs = batch_data['x_lm_input'].to(device=device)
        x_lm_attention = batch_data['x_lm_attention'].to(device=device)
        x_lm_type_ids = batch_data['x_lm_type_ids'].to(device=device)
        x_lm_span = batch_data['x_lm_spans'].to(device=device)

        y_all_preds = net(x_word, x_char, x_lm_inputs, x_lm_attention, x_lm_type_ids, x_lm_span, masks)
        
        y_all_targets = []
        
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
