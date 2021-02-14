from .. import *
from ..preprocess import *
from ..preprocess.genia import *

def run_preprocess_genia(logger, filepath, cased, lm_name, no_entity='O'):
    logger.info('== GENIA PREPROCESS ==')

    logger.info('Loading tokenizer')
    tokenizer = get_tokenizer(lm_name=lm_name, lowercase=(not cased))

    logger.info('Preprocessing raw data...')
    genia_data = []
    root = ET.parse(filepath).getroot()
    
    for child in root:
        if child.tag != 'article':
            continue

        child_data = get_raw_entities(child.find('title').findall('sentence'))
        child_data += get_raw_entities(child.find('abstract').findall('sentence'))

        for c in child_data:
            c['entities'] = expand_nested_attributes(c['entities'])
            c['entities'] = get_norm_entities(c['entities'])
            
            c['text'] = c['tokens']
            c['tokens'], c['entities'], c['token_offsets'] = transform_text_spans(c['tokens'], c['entities'],
                                                              tokenizer, lowercase=(not cased))

        genia_data += child_data
    
    # Obtain dictionary of entity types and add an ID to each item
    logger.info('Getting dictionary of entities...')
    genia_et_freq = {}
    for i, item in enumerate(genia_data):
        genia_data[i]['item_id'] = i
        for e in item['entities']:
            if e['entity_type'] not in genia_et_freq:
                genia_et_freq[e['entity_type']] = 1
            else:
                genia_et_freq[e['entity_type']] += 1

    # Sanity check
    logger.info('Running sanity check...')
    n_bad = sanity_check(genia_data)
    if n_bad == 0:
        logger.info('Sanity check passed')
    else:
        logger.warning('Sanity check found %d errors' % n_bad)
        raise Exception('Sanity check found %d errors' % n_bad)

    # Split and store
    logger.info('Split and store dataset...')
    train_dataset, dev_dataset, test_dataset = split_dataset(genia_data)
    store_datasets('genia', train_dataset, dev_dataset, test_dataset)
    
    logger.info('Done')
