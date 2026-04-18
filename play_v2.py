import os
import sys

# --- WSL2 GRAPHICS FIX (Apply before any other imports) ---
os.environ['MUJOCO_GL'] = 'egl' 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import copy
import time
import mujoco
import mujoco.viewer
# --- NEW IMPORT ---
from stable_baselines3.common.env_util import make_vec_env

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
    
    print("[INFO] Initializing Single Physical Body for Playback...")
    # Wrap the env in make_vec_env so the Bridge finds .num_envs and batched data
    vec_env = make_vec_env(
        lambda: BerkeleyHumanoidMujocoEnvV2(xml_path=xml_path, render_mode=None),
        n_envs=1
    )

    # Re-extract the underlying MuJoCo attributes for the viewer to use
    base_env = vec_env.envs[0].unwrapped
    
    # Initialize the Bridge with the vectorized wrapper
    env = RSLRL_Bridge(vec_env, device="cpu")
    
    print("[INFO] Resetting environment...")
    obs = env.reset()

    # --- 2. PREPARE CONFIGURATION ---
    train_cfg_dict = class_to_dict(BerkeleyCfg)
    train_cfg_dict["obs_groups"] = BerkeleyCfg.obs_groups
    train_cfg_dict["num_steps_per_env"] = BerkeleyCfg.num_steps_per_env
    train_cfg_dict["save_interval"] = BerkeleyCfg.runner.save_interval
    train_cfg_dict["experiment_name"] = BerkeleyCfg.runner.experiment_name
    train_cfg_dict["run_name"] = BerkeleyCfg.runner.run_name

    # --- 3. INITIALIZE RUNNER & LOAD POLICY ---
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device="cpu")
    
    print(f"[INFO] Loading model weights from {checkpoint_path}...")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device="cpu")
    # No need for policy.eval() here as get_inference_policy handles it

    # --- 4. RENDER LOOP ---
    print("[INFO] Launching MuJoCo Viewer. Press ESC to close.")
    
    # Lift the robot initially
    base_env.data.qpos[2] = 0.9
    mujoco.mj_forward(base_env.model, base_env.data)

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            
            with torch.no_grad():
                # Pass the actor observations (the 'obs' key) to the policy
                actions = policy(obs)
            
            # The bridge now returns 4 items because of the VecEnv wrapper
            obs, rewards, dones, infos = env.step(actions)
            
            # Reset logic (commented out as per your previous version)
            # if dones.any():
            #     obs = env.reset()

            try:
                viewer.sync()
            except Exception:
                break
            
            # Control playback speed (50Hz)
            elapsed = time.time() - start_time
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    # Check for the model file (Updated path to reflect Robert's setup if needed)
    model_to_load = os.path.join(current_dir, "logs/berkeley_humanoid_asym/model_49.pt")
    
    if os.path.exists(model_to_load):
        play(model_to_load)
    else:
        print(f"[ERROR] Could not find {model_to_load}")
        log_dir = os.path.join(current_dir, "logs/berkeley_humanoid_asym/")
        if os.path.exists(log_dir):
            print("Available models in log folder:", os.listdir(log_dir))