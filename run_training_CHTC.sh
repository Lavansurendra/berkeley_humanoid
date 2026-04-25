#!/bin/bash

# 1. Tell W&B to stay offline (Crucial for CHTC nodes)
export WANDB_MODE=offline
# 2. Tell W&B where to save the logs so they get transferred back to you
export WANDB_DIR=./output

# 3. Unpack your environment (Assuming you've tarred your conda env)
# This part depends on how you package your dependencies
tar -xzf mujoco_env.tar.gz
export PATH=$PWD/mujoco_env/bin:$PATH

# 4. Run the training
python train.py