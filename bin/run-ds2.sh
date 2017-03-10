#!/bin/sh

set -xe

export ds_importer="ted"
export ds_dataset_path="/data/LIUM"

export ds_train_batch_size=32
export ds_dev_batch_size=32
export ds_test_batch_size=32

export ds_learning_rate=0.0004

export ds_epochs=20
export ds_display_step=10
export ds_validation_step=5
export ds_checkpoint_step=1

# export ds_limit_train=1000
# export ds_limit_dev=100
# export ds_limit_test=100

# export ds_export_dir="${ds_dataroot}/exports/`git rev-parse --short HEAD`"

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py
