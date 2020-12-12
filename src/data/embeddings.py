import numpy as np

def load_embedding_matrix(filepath, dimension, special_tokens):
    id2word = []
    word2id = {}
    
    # Read GloVe vectors
    glove = {}
    with open(filepath, 'rb') as f:
        for line in f:
            values = line.decode().split()
            word = values[0]
            
            word2id[word] = len(id2word)
            id2word.append(word)
            glove[word] = values[1:]

    # Build embedding matrix
    embedding_matrix = np.zeros((len(glove) + len(special_tokens), dimension), dtype=np.float)
    
    for idx, word in enumerate(id2word):
        embedding_matrix[idx] = np.asarray(glove[word], dtype=np.float)
    
    # Add special tokens and randomly initialize them
    for special_token in special_tokens:
        token_id = len(id2word)
        word2id[special_token] = token_id
        id2word.append(special_token)
        embedding_matrix[token_id] = np.random.normal(size=dimension)
    
    return embedding_matrix, id2word, word2id
