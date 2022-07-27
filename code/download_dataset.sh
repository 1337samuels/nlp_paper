#!/bin/bash
KAGGLE_KEY='/root/.kaggle/kaggle.json'
if [[ ! -f $KAGGLE_KEY ]]; then
  printf '%s does not exist!\n' $KAGGLE_KEY >&2
  exit 1
fi

chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d Cornell-University/arxiv

mkdir -p optagan/data/arxiv_data/
mkdir -p ../input/arxiv
unzip arxiv.zip -d ../input/arxiv
rm arxiv.zip

python create_test_data.py --category cs --arxiv-data-file ../input/arxiv/arxiv-metadata-oai-snapshot.json --output-dir optagan/data/arxiv_data/
