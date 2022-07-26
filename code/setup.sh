#!/bin/bash

pip install -r requirements.txt
git clone https://github.com/Egojr/optagan.git

./download_dataset.sh

mkdir -p optagan/data/arxiv_data/
mkdir - p ../input/arxiv
unzip arxiv.zip - d ../input/arxiv
rm arxiv.zip

python create_test_data.py --category cs.LG --arxiv-data-file ../input/arxiv/arxiv-metadata-oai-snapshot.json --output-dir optagan/data/arxiv_data/

./download_model.sh