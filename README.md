# Name TBD

This repository contains the code of the paper: [Name TBD](https://www.overleaf.com/project/62d6c596e95fd04807c23c2b)


### Installation
We trained and evaluated the model in Google Colab. We provided scripts in this project in order to ease the process of running the model without going through an entire notebook.

To reproduce the results found in our paper please follow the instructions:
#### Prerequisites

#### Setup
First set your chosed parameters:
```python
LATENT_SIZE=32
BETA=0
BLOCK_SIZE=150
CHECKPOINT_DIR='./arxiv_model'
OUTPUT_DIR='./gandirectory'
TRAIN_FILE='optagan/data/arxiv_data/train.txt'
EVAL_FILE='optagan/data/arxiv_data/test.txt'
```
Download the dataset and Optimus model which will later be fine-tuned. For our model we chose only papers in Computer Science and filtered out those over 150 words, leaving us with aroung 120 thousand papers.
```sh
./download_dataset.sh
LATENT_SIZE=$LATENT_SIZE ./download_model.sh
```
Train the model, note that the `gloabl_step_eval` parameter in the GAN training part of the model is a function of the number of training examples and batch size.
```sh
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
    --checkpoint_dir=./optimus_model/checkpoint-508523 \
    --latent_size $LATENT_SIZE \
    --gloabl_step_eval  508523 \
    --do_lower_case
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
    --gloabl_step_eval 10966 \
    --latent_size $LATENT_SIZE \
    --block_dim 100 \
    --n_layers 10 \
    --interval 50 \
    --epochs 4 \
    --finetune_decoder True \
    --lr_rl 1e-6 \
    --epochs_rl 300 \
    --batch_size_rl 32 \
    --do_lower_case
```
### Evaluation
We haven't yet developed a formal metric for measuring the creativity or viability of the generated abstracts. However for our experiments we used the following script:
```sh
cp test_dimensions.py optagan/optagan # For local imports
python optagan/optagan/test_dimensions.py \
    --checkpoint_dir=$OUTPUT_DIR \
    --output_dir=./new_output \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_eval_batch_size=128 \
    --gloabl_step_eval 10966 \
    --block_size $BLOCK_SIZE \
    --max_seq_length 100 \
    --latent_size $LATENT_SIZE
```
* Note that this can be run for one dimension or multiple.
To calcualte the latent space of all the vectors in the test set (can be used for later analysis):
```sh
mkdir ./new_output
cp  calculate_latent_space.py optagan/optagan # For local imports
python optagan/optagan/calculate_latent_space.py \
    --checkpoint_dir=$OUTPUT_DIR \
    --output-file=$OUTPUT_FILE \
    --encoder_model_type=bert \
    --encoder_model_name_or_path=bert-base-cased \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=gpt2 \
    --block_size 100 \
    --max_seq_length 100 \
    --latent_size $LATENT_SIZE \
    --global_step_eval 10966 \
    --eval_data_file=$TEST_FILE
```
### Generation
Although we mainly use the VAE, we have also trained a GAN so we can generate new abstracts (which are pretty believable). we can use the following generation code from the original Optagan paper.
```sh
! python optagan/optagan/wgan_test.py \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --output_dir=output \
    --generator_dir=$OUTPUT_DIR \
    --block_size 100 \
    --max_seq_length 60 \
    --gloabl_step_eval 10990 \
    --latent_size $LATENT_SIZE \
    --block_dim 100 \
    --new_sent 1000 \
    --n_layers 10 \
    --top_p 0.9 \
    --output_name=results \
    --save True
! cat output/results.txt
```

### References

We adapt most of our code from the [Optagan repository](https://github.com/Egojr/optagan)
