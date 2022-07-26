#!/bin/bash

mkdir ./myoutput
python optagan/optagan/wgan_test.py \
    --checkpoint_dir=./arxiv_model \
    --output_dir=./myoutput \
    --generator_dir=./gandirectory \
    --block_size $BLOCK_SIZE \
    --max_seq_length 60 \
    --gloabl_step_eval 2330 \
    --latent_size $LATENT_SIZE \
    --block_dim 100 \
    --new_sent 1000 \
    --n_layers 10 \
    --top_p 0.9 \
    --output_name=results \
    --save True