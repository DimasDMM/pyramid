#!/bin/bash

source activate tf

python run_evaluator.py \
    --model_ckpt ./artifacts/genia/ \
    --dataset genia \
    --device cuda \
    --log_to_file 0
