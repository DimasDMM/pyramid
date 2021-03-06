#!/bin/bash

source activate tf

python run_training.py \
    --model_ckpt ./artifacts/genia/ \
    --wv_file ./data/glove.6B.200d.txt \
    --use_char_encoder 1 \
    --dataset genia \
    --max_epoches 500 \
    --total_layers 16 \
    --batch_size 64 \
    --token_emb_dim 200 \
    --char_emb_dim 100 \
    --cased_lm 1 \
    --cased_word 0 \
    --cased_char 0 \
    --hidden_dim 100 \
    --dropout 0.45 \
    --lm_name dmis-lab/biobert-large-cased-v1.1 \
    --lm_emb_dim 1024 \
    --device cuda \
    --continue_training 0 \
    --log_to_file logger_genia.txt
