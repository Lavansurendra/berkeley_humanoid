import os
import sys
import torch
import copy

# --- THE BORROWED PLUMBING ---
from stable_baselines3.common.env_util import make_vec_env

# 1. PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Ensure the exts folder is in the system path for imports
sys.path.append(os.path.join(current_dir, "exts/berkeley_humanoid"))

from rsl_rl.runners import OnPolicyRunner
from berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env_v2 import BerkeleyHumanoidMujocoEnvV2
from rsl_bridge import RSLRL_Bridge
from config import BerkeleyCfg

def class_to_dict(obj):
    """Recursively converts classes to dicts with deepcopy protection."""
    result = {}
    for key in dir(obj):
        if key.startswith("_"): continue
        element = getattr(obj, key)
        if isinstance(element, type):
            result[key] = class_to_dict(element)
        elif isinstance(element, dict):
            result[key] = copy.deepcopy(element)
        elif not callable(element):
            result[key] = element
    return result

def main():
    # --- 1. SETUP ASSETS ---
    xml_path = os.path.join(current_dir, "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml")

    # --- VECTORIZED PHYSICAL BODIES ---
    print(f"[INFO] Initializing {BerkeleyCfg.env.num_envs} Parallel Physical Bodies (V2 Physics)...")
    # We use SB3 here to spin up the 8 parallel CPU threads
    vec_env = make_vec_env(
        lambda: BerkeleyHumanoidMujocoEnvV2(xml_path=xml_path, render_mode=None),
        n_envs=BerkeleyCfg.env.num_envs
    )

    print("[INFO] Initializing Sensory Bridge (Asymmetric Layer)...")
    # We pass the vectorized bundle to the bridge instead of a single env
    env = RSLRL_Bridge(vec_env, device="cpu")
    env.reset()

    # --- 2. PREPARE CONFIGURATION ---
    print("[INFO] Converting Configuration...")
    train_cfg_dict = class_to_dict(BerkeleyCfg)
    
    # Flatten/Hoist critical keys for the modular rsl_rl runner
    train_cfg_dict["obs_groups"] = BerkeleyCfg.obs_groups
    train_cfg_dict["num_steps_per_env"] = BerkeleyCfg.num_steps_per_env
    train_cfg_dict["save_interval"] = BerkeleyCfg.runner.save_interval
    train_cfg_dict["experiment_name"] = BerkeleyCfg.runner.experiment_name
    train_cfg_dict["run_name"] = BerkeleyCfg.runner.run_name

    log_dir = os.path.join(current_dir, "logs", BerkeleyCfg.runner.experiment_name)
    
    # --- 3. RUN TRAINING ---
    print("[INFO] Setting up modular rsl_rl Runner...")
    runner = OnPolicyRunner(
        env, 
        train_cfg_dict, 
        log_dir=log_dir, 
        device="cpu"
    )

    print("[INFO] Starting Training...")
    runner.learn(num_learning_iterations=BerkeleyCfg.runner.max_iterations)
    
    print("[INFO] Training Complete. System Exiting.")

if __name__ == "__main__":
    main()