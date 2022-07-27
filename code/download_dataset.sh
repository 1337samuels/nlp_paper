#!/bin/bash
KAGGLE_KEY='/root/.kaggle/kaggle.json'
if [[ ! -f $KAGGLE_KEY ]]; then
  printf '%s does not exist!\n' $KAGGLE_KEY >&2
  exit 1
fi

chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d Cornell-University/arxiv