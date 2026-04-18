import os
import sys

# --- WSL2 GRAPHICS FIX (Apply before any other imports) ---
# 'egl' is the standard for WSL2/WSLg. 
# If it stays black, change this to 'osmesa' for software rendering.
os.environ['MUJOCO_GL'] = 'egl' 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import copy
import time
import mujoco
import mujoco.viewer

# 1. PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
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

def play(checkpoint_path):
    # --- 1. SETUP ENVIRONMENT ---
    xml_path = os.path.join(current_dir, "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml")
    
    # We use 'human' render mode for the window
    base_env = BerkeleyHumanoidMujocoEnvV2(xml_path=xml_path, render_mode=None)
    env = RSLRL_Bridge(base_env, device="cpu")
    
    # Populate observations so the runner can 'see' the sensors during init
    print("[INFO] Resetting environment...")
    obs = env.reset()

    # --- 2. PREPARE CONFIGURATION ---
    train_cfg_dict = class_to_dict(BerkeleyCfg)
    # Flatten config for modular rsl_rl
    train_cfg_dict["obs_groups"] = BerkeleyCfg.obs_groups
    train_cfg_dict["num_steps_per_env"] = BerkeleyCfg.num_steps_per_env
    train_cfg_dict["save_interval"] = BerkeleyCfg.runner.save_interval
    train_cfg_dict["experiment_name"] = BerkeleyCfg.runner.experiment_name
    train_cfg_dict["run_name"] = BerkeleyCfg.runner.run_name

    # --- 3. INITIALIZE RUNNER & LOAD POLICY ---
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device="cpu")
    
    print(f"[INFO] Loading model weights...")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device="cpu")
    policy.eval()

    # --- 4. RENDER LOOP ---
    print("[INFO] Launching MuJoCo Viewer. Press ESC to close.")
    
    # Lift the robot initially
    base_env.data.qpos[2] = 0.9
    mujoco.mj_forward(base_env.model, base_env.data)

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            
            with torch.no_grad():
                actions = policy(obs)
            
            obs, rewards, dones, infos = env.step(actions)
            
            # # If the robot falls (z < 0.3) or time runs out, reset
            # if dones.any():
            #     obs = env.reset()
            #     base_env.data.qpos[2] = 0.9 # Re-lift
            #     mujoco.mj_forward(base_env.model, base_env.data)

            # Try-except prevents the Segfault when closing the window
            try:
                viewer.sync()
            except Exception:
                break
            
            # Control playback speed (50Hz)
            elapsed = time.time() - start_time
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    # Check for the model file
    model_to_load = os.path.join(current_dir, "logs/berkeley_humanoid_asym/model_49.pt")
    
    if os.path.exists(model_to_load):
        play(model_to_load)
    else:
        print(f"[ERROR] Could not find {model_to_load}")
        log_dir = os.path.join(current_dir, "logs/berkeley_humanoid_asym/")
        if os.path.exists(log_dir):
            print("Available models in log folder:", os.listdir(log_dir))