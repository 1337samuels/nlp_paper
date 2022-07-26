#!/bin/bash
 mkdir output
# interpolation    
python optagan/optagan/run_latent_generation.py \
    --checkpoint_dir=./arxiv_model \
    --output_dir=./output \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --per_gpu_eval_batch_size=128 \
    --gloabl_step_eval 457 \
    --block_size $BLOCK_SIZE \
    --max_seq_length 100 \
    --latent_size 32 \
    --interact_with_user_input \
    --play_mode interpolation \
    --sent_source="using the technique of the metrization theorem of uniformities with countable bases ,  in this note we provide ,  test and compare an explicit algorithm to produce a metric of an affinity weighted undirected graph ." \
    --sent_target="we provide a simple and efficient algorithm for the projection operator for weighted  - norm regularization subject to a sum constraint ,  together with an elementary proof .  the implementation of the proposed algorithm can be downloaded from the author ' s homepage ." \
    --num_interpolation_steps=10