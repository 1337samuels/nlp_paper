#!/bin/bash

if [[ ! -f '/root/.kaggle/kaggle.json' ]] then
  printf '%s does not exist!\n' " '" >&2
  exit 1
fi

chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download - d Cornell-University/arxiv