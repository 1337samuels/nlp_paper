#!/bin/bash

python optagan/optagan/run_lm_vae_training.py \
    --output_dir=$CHECKPOINT_DIR \
    --dataset EMNLP \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --beta $BETA \
    --ratio_zero 0.5 \
    --ratio_increase 0.25 \
    --do_train \
    --fb_mode 0 \
    --dim_target_kl 0.5\
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --num_train_epochs 1.0 \
    --save_steps 10000 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=10 \
    --block_size $BLOCK_SIZE \
    --length_weighted_loss \
    --use_pretrained_model \
    --checkpoint_dir=./optimus_model \
    --latent_size $LATENT_SIZE \
    --gloabl_step_eval  508523 \
    --do_lower_case

mkdir $OUTPUT_DIR
python optagan/optagan/optagan.py \
    --dataset EMNLP \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --output_dir=$OUTPUT_DIR \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --valid_data_file=$EVAL_FILE \
    --per_gpu_train_batch_size 128 \
    --block_size $BLOCK_SIZE \
    --max_seq_length 50 \
    --gloabl_step_eval 2330 \
    --latent_size $LATENT_SIZE \
    --block_dim 100 \
    --n_layers 10 \
    --interval 50 \
    --epochs 50 \
    --finetune_decoder True \
    --lr_rl 1e-6 \
    --epochs_rl 200 \
    --batch_size_rl 32 \
    --do_lower_case
