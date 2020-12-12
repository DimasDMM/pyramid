
def add_layer_outputs(dataset, total_layers=16, entity_dict=None):
    if entity_dict is None:
        init_entity_dict = True
        entity_dict = {'O': 0}
    else:
        init_entity_dict = False
    
    # Create dictionary of entities
    for i, item in enumerate(dataset):
        layer_seq = [0] * len(item['tokens'])
        dataset[i]['layer_outputs'] = [layer_seq[i_layer:] for i_layer in range(total_layers)]
        if init_entity_dict:
            for entity in item['entities']:
                entity_type = entity['entity_type']
                b_entity_type = 'B-%s' % entity_type
                i_entity_type = 'I-%s' % entity_type
                if b_entity_type not in entity_dict:
                    entity_dict[b_entity_type] = len(entity_dict)
                if i_entity_type not in entity_dict:
                    entity_dict[i_entity_type] = len(entity_dict)

    # Generate outputs of each layer
    last_layer = total_layers - 1
    for item in dataset:
        for entity in item['entities']:
            span_start = entity['span'][0]
            span_end = entity['span'][1]
            b_type_id = entity_dict['B-%s' % entity['entity_type']]
            i_type_id = entity_dict['I-%s' % entity['entity_type']]

            length = span_end - span_start
            if length >= total_layers:
                item['layer_outputs'][last_layer][span_start] = b_type_id
                item['layer_outputs'][last_layer][span_start+1:span_end+1] = [i_type_id]*(length)
            else:
                item['layer_outputs'][length][span_start] = b_type_id
    
    return dataset, entity_dict

def build_char_vocab(genia_data, lower_case=False, special_tokens=[]):
    id2char = []
    for item in genia_data:
        if lower_case:
            item_chars = [x.lower() for x in ''.join(item['tokens'])]
        else:
            item_chars = [x for x in ''.join(item['tokens'])]
        id2char += item_chars
    
    # Remove duplicates and generate inverse dictionary
    id2char = list(set(id2char))
    char2id = {x:i for i, x in enumerate(id2char)}
    
    # Add special tokens
    for special_token in special_tokens:
        token_id = len(id2char)
        char2id[special_token] = token_id
        id2char.append(special_token)
    
    return id2char, char2id