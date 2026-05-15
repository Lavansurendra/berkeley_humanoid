#!/bin/bash

echo "=================================================="
echo "CHTC START TIME: $(date)"
echo -n "IMAGE BUILD DATE: "
# This file is created in your Dockerfile during the 'v4' build
cat /image_build_date.txt || echo "ERROR: Build date file not found."
echo "=================================================="

# Define the path to the code inside your Docker image
CODE_DIR="/workspace"

# These lines "trick" PyTorch into thinking it knows who you are
export USER=trainer
export LOGNAME=trainer
export HOME=/tmp

# This tells PyTorch where to put its CPU-math cache
export TORCHINDUCTOR_CACHE_DIR=./output/torch_cache

# W&B settings
export WANDB_MODE=offline
export WANDB_DIR=./output
export WANDB_DISABLE_SYMLINKS=true

# Create the output directory immediately so HTCondor never fails the transfer
mkdir -p output

echo "Running training script with the following settings:"

echo "Environment has the below packages:"
pip list

echo "Starting training (train.py)..."
python3 $CODE_DIR/train.py "$@"

find output/wandb -type l -delete

echo "Training complete!"
