# Wiscohumanoids: Environment Setup & Execution Guide

Welcome to the **Wiscohumanoids** repository! This project focuses on training bipedal robots (the Berkeley Humanoid) to walk using reinforcement learning (Stable Baselines3 PPO) within a standalone MuJoCo physics simulation.

This step-by-step guide is designed to take a new researcher from a completely clean OS installation to a fully functional, cloud-monitored training environment that is ready for deployment on local machines or the CHTC cluster.

---

## Operating System & Linux Prerequisites

Because high-performance physics engines (MuJoCo) and RL libraries are native to Linux, Windows users must use **WSL2 (Windows Subsystem for Linux)**.

### 1. Install WSL2 (Windows Users Only)
Open PowerShell as an Administrator and run:

<pre>
  powershell
  wsl --install
</pre>

Restart your computer when this is done

### 2. Install Linux Build Essentials
Open your Ubuntu terminal. Before installing Python packages, the OS needs the C++ compilers and OpenGL libraries required to build the simulation environments. Run:

<pre>
  sudo apt update && sudo apt upgrade -y
  sudo apt install build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf -y
</pre>


## Python & Conda Environment Setup
### 1. Install Miniconda
In your Ubuntu terminal run the following commands

<pre>
  mkdir -p ~/miniconda3
  wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm -rf ~/miniconda3/miniconda.sh
  ~/miniconda3/bin/conda init bash
</pre>

### 2. Create a conda environment for this project
NOTE: the environment name (the word that follows -n) in the following command can be customized to whatever you choose

<pre>
  conda create -n mujoco python=3.10 -y
  conda activate mujoco
</pre>

If this works correctly you should see in brackets the name of your conda environment at the start of your next line in the terminal

### 3. Install dependencies
The following is all the dependencies you will need to have installed in order to run the code. Ensure you have your conda environment active when you run the following line:

<pre>
  pip install -r setup/requirements.txt
</pre>

This installs RL algorithms, Gymnasium, and MuJoCo bindings along w/ monitoring / logging utilities. Check [`setup/requirements.txt`](setup/requirements.txt) for the most up-to-date list.


### 4. Cloud Monitoring (Weights & Biases)
We use Weights & Biases (W&B) to track rewards, hardware utilization (CPU/RAM), and to save model checkpoints remotely.
    1. Create a free account at wandb.ai.
    2. Link your terminal to your account:

<pre>
  wandb login
</pre>


### 5. Execution

There are two ways in which a training can be executed:

#### Locally

To kick off a training run first you must set the num_envs (which you can set to half the number of threads on your cpu) and total_timesteps in the train.py file. Then, with your conda environment active run:

<pre>
  python train.py
</pre>


To see your results you will need to run:
<pre>
  python play.py
</pre>

NOTE: Make sure that the model_name and stats_path variable correspond to the model you are running

#### CHTC

