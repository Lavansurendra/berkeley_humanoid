#!/bin/bash

# 1. Tell W&B to stay offline (Crucial for CHTC nodes)
export WANDB_MODE=offline
# 2. Tell W&B where to save the logs so they get transferred back to you
export WANDB_DIR=./output

# 4. Run the training
pip list

python train.py

#DUMMY output!