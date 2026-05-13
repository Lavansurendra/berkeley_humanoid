#!/bin/bash

# W&B settings
export WANDB_MODE=offline
export WANDB_DIR=./output

echo "Running training script with the following settings:"

echo "Environment has the below packages:"
pip list

echo "Starting training (train.py)..."
python train.py

echo "Training complete!"