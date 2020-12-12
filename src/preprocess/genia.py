import xml.etree.ElementTree as ET
import re

from ..data.tokenization import *

def get_inner_xml(element):
    return (element.text or '') + ''.join(ET.tostring(e, 'unicode') for e in element)

def parse_cons_attributes(match):
    attributes = {}
    for item in re.findall(r'[a-z]+="[^"]+"\s*', match):
        m = re.search(r'^([^=]+)="([^"]+)"', item.strip())
        name = m[1]
        value = m[2]
        attributes[name] = value
    return attributes

def reverse_enumerate(L):
    i = len(L)
    while i > 0:
        i -= 1
        yield i, L[i]

def get_raw_entities(xml_sentences):
    data = []
    regex_open = r'<cons\s*((?:[a-z]+="[^"]+"\s*)*)>'
    regex_close = r'</cons>'
    
    for sentence in xml_sentences:
        entities = []
        inner_xml = get_inner_xml(sentence)

        m_open = re.search(regex_open, inner_xml)
        m_close = re.search(regex_close, inner_xml)
        while m_open is not None or m_close is not None:
            # Check which regex matched first
            if m_close is None or m_open is not None and m_open.span()[0] < m_close.span()[0]:
                inner_xml = re.sub(regex_open, '', inner_xml, 1)
                cons_attributes = parse_cons_attributes(m_open.group(1))
                entities.append({'span': [m_open.span()[0], -1], 'attributes': cons_attributes})
            else:
                inner_xml = re.sub(regex_close, '', inner_xml, 1)
                for _, e in reverse_enumerate(entities):
                    # Add close_span to the latest non-closed entity
                    if e['span'][1] == -1:
                        span_start = e['span'][0]
                        e['span'] = [span_start, m_close.span()[0]]
                        break
                            
            m_open = re.search(regex_open, inner_xml)
            m_close = re.search(regex_close, inner_xml)

        data.append({'tokens': inner_xml, 'entities': entities})
    
    return data

def split_sem_lex_info(lexsem_types):
    lexsem_types = re.sub(r'^\((AND|OR|AND/OR|AS\_WELL\_AS|BUT\_NOT) ', '', lexsem_types)
    lexsem_types = re.sub(r'\)$', '', lexsem_types)
    lexsem_types = lexsem_types.strip().split()
    return lexsem_types

def expand_nested_attributes(entities):
    """
    Expands 'sem' attribute from the parent entity to the children.
    """
    for e_parent in entities:
        if 'sem' not in e_parent['attributes'] or not re.search(r'^\(AND', e_parent['attributes']['sem']):
            continue
        
        lex_types = split_sem_lex_info(e_parent['attributes']['lex'])
        sem_types = split_sem_lex_info(e_parent['attributes']['sem'])
        
        for i, lex_type in enumerate(lex_types):
            for _ in re.findall(r',', lex_type):
                sem_types = sem_types[:i] + [sem_types[i]] + sem_types[i:]
        
        i_sem = 0
        for e_child in entities:
            if 'sem' in e_child['attributes']:
                continue
            elif e_parent['span'][0] > e_child['span'][0] or e_parent['span'][1] < e_child['span'][1]:
                continue
            elif re.search(r'^\*', e_child['attributes']['lex']):
                continue
            
            if len(sem_types) > i_sem:
                e_child['attributes']['sem'] = sem_types[i_sem]
            else:
                e_child['attributes']['sem'] = 'G#other_name'

            i_sem += 1
        
    return entities

def get_entity_type(sem):
    if re.search(r'^G\#other', sem):
        return None
    elif re.search(r'^G\#RNA', sem):
        return 'RNA'
    elif re.search(r'^G\#DNA', sem):
        return 'DNA'
    elif re.search(r'^G\#protein', sem):
        return 'protein'
    elif re.search(r'^G\#cell\_line', sem):
        return 'cell_line'
    elif re.search(r'^G\#cell\_type', sem):
        return 'cell_type'
    else:
        return None
        #raise Exception('Unknown sem: %s' % sem)

    return sem

def get_norm_entities(entities):
    """
    Apply these two preprocess steps:
    - Collapses all DNA, RNA, and protein subtypes into DNA, RNA, and protein, keeping cell line and cell type.
    - Removes other entity types, resulting in 5 entity types.
    """
    
    i = len(entities)
    while i > 0:
        i -= 1
        e = entities[i]

        if 'sem' not in e['attributes']:
            del entities[i]
            continue
        
        if re.search(r'^\((AND|OR|AND/OR|AS\_WELL\_AS|BUT\_NOT) ', e['attributes']['sem']):
            # Parent entity
            sem_types = e['attributes']['sem']
            sem_types = re.sub(r'^\([^\s]+', '', sem_types)
            sem_types = re.sub(r'\)$', '', sem_types)
            sem = sem_types.strip().split()[0]
        else:
            sem = e['attributes']['sem']
        
        entity_type = get_entity_type(sem)
        if entity_type is None:
            # Remove this entity
            del entities[i]
        else:
            del e['attributes']
            e['entity_type'] = entity_type

    return entities

def transform_text_spans(text, entity_list, tokenizer, lowercase=True):
    tokens, spans = tokenize_text(tokenizer, text, lowercase=lowercase)
    for i, entity in enumerate(entity_list):
        span_start, span_end = entity['span']
        new_span_start = -1
        new_span_end = -1
        
        j = 0
        while j < len(spans):
            token_span = spans[j]
            if span_start > token_span[0] and span_start < token_span[1]:
                # Split token
                tokens = tokens[:j] + [tokens[j][:span_start], tokens[j][span_start:]] + tokens[(j+1):]
                spans = spans[:j] + [[token_span[0], span_start-1], [span_start, token_span[1]]] + spans[(j+1):]
                token_span = spans[j]
            
            if span_end > token_span[0] and span_end < token_span[1]:
                # Split token
                tokens = tokens[:j] + [tokens[j][:span_end], tokens[j][span_end:]] + tokens[(j+1):]
                spans = spans[:j] + [[token_span[0], span_end-1], [span_end, token_span[1]]] + spans[(j+1):]
                token_span = spans[j]

            if span_start == token_span[0]:
                new_span_start = j
            elif span_start == token_span[1]:
                new_span_start = j + 1
            elif span_start > token_span[1] and len(spans) > j+1 and span_start < spans[j+1][0]:
                new_span_start = j + (span_start - token_span[1])

            if span_end < token_span[1] and span_end > token_span[0]:
                new_span_end = j - (token_span[1] - span_end)
                break
            elif span_end > token_span[1] and len(spans) > j+1 and span_end < spans[j+1][0]:
                new_span_end = j + (span_end - token_span[1])
                break
            elif span_end == token_span[0]:
                new_span_end = j - 1
                break
            elif span_end == token_span[1]:
                new_span_end = j
                break
            elif span_end > token_span[1] and len(spans) == j+1:
                new_span_end = j
                break
            
            j += 1
        
        if new_span_start != -1 and new_span_end != -1:
            entity['span'] = [new_span_start, new_span_end]

    return tokens, entity_list
