#!/bin/bash

pip install -r requirements.txt
git clone https://github.com/Egojr/optagan.git

./download_dataset.sh

./download_model.sh