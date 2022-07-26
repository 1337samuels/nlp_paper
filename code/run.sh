#!/bin/bash

LATENT_SIZE=${LATENT_SIZE:=32}
BETA=${BETA:=0}
BLOCK_SIZE=${BLOCK_SIZE:=100}
CHECKPOINT_DIR=${CHECKPOINT_DIR:=./arxiv_model}
OUTPUT_DIR=${OUTPUT_DIR:=./gandirectory}
TRAIN_FILE=${TRAIN_FILE:=optagan/data/arxiv_data/train.txt}
EVAL_FILE=${EVAL_FILE:=optagan/data/arxiv_data/test.txt}

if [[ -d "$OUTPUT_DIR" ]]
then
    mkdir "$OUTPUT_DIR"
fi


if [[ -d "$CHECKPOINT_DIR" ]]
then
    mkdir "$CHECKPOINT_DIR"
fi

./setup.sh
./train.sh