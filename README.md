# Pyramid

## Introduction

TODO

## Set up

Clone this repository, create default folders and install dependencies:
```sh
git clone https://github.com/DimasDMM/pyramid.git
cd pyramid
mkdir data
mkdir artifacts
pip install -r requirements.txt
```

Download GloVe embeddings:
```sh
cd data
wget http://nlp.stanford.edu/data/glove.6B.zip --no-check-certificate
unzip glove.6B.zip
cd ..
```

It is necessary that you download the tokenizer and pretrained LM beforehand:
```sh
python run_download.py --lm_name dmis-lab/biobert-v1.1 --log_to_file 0
```

### GENIA dataset

Download and preprocess the GENIA dataset:
```sh
cd data
wget http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz --no-check-certificate
mkdir GENIA
tar -xvf GENIAcorpus3.02.tgz -C GENIA
cd ..
python run_preprocess.py \
    --dataset genia \
    --raw_filepath "./data/GENIA/GENIA_term_3.02/GENIAcorpus3.02.xml" \
    --lm_name dmis-lab/biobert-v1.1 \
    --cased 0 \
    --log_to_file 0
```

## Commands

Run model training:
```sh
python run_training.py \
    --model_ckpt ./artifacts/genia/ \
    --wv_file ./data/glove.6B.100d.txt \
    --dataset genia \
    --max_epoches 100 \
    --eval_on_training 0 \
    --total_layers 16 \
    --batch_size 64 \
    --token_emb_dim 100 \
    --char_emb_dim 100 \
    --cased 0 \
    --hidden_dim 100 \
    --dropout 0.4 \
    --freeze_wv 1 \
    --lm_name dmis-lab/biobert-v1.1 \
    --lm_emb_dim 768 \
    --device cuda \
    --log_to_file 0
```

Run evaluation in a trained model:
```sh
python run_training.py \
    --model_ckpt ./artifacts/genia/ \
    --dataset genia \
    --device cuda \
    --log_to_file 0
```
