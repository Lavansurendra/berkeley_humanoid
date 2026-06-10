import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment.berkeley_env import BerkeleyEnv
from environment.g1_env import G1Env



def main():
    # 1. Update the XML path to the new assets folder
    xml_path = "environment/g1_scene.xml"

    # 2. Update model paths to point to the new output/models directory
    models_dir = "output/models"
    if len(sys.argv) > 0:
       model_name = sys.argv[0] 
    else : 
        print('[ERROR] missing model name') 
        return    
 
    model_path = os.path.join(models_dir, model_name) # No .zip needed for PPO.load
    stats_path = os.path.join(models_dir, "g1_vecnormalize_200000.pkl")
    
    print(f"[INFO] Loading MuJoCo Environment...")
    
    # Wrap in G1Env and then DummyVecEnv for compatibility with Stable Baselines3
    base_env_fn = lambda: G1Env(xml_path=xml_path, render_mode="human")
    env = DummyVecEnv([base_env_fn])
    
    # Load and freeze normalization stats
    if os.path.exists(stats_path):
        print(f"[INFO] Loading normalization statistics from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False 
        env.norm_reward = False 
    else:
        print(f"[WARNING] Could not find {stats_path}. Performance may be poor!")
    
    # Check for model existence (PPO.load expects path without .zip but os.path needs it)
    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Could not find {model_path}.zip. Check your output/models folder!")
        return

    print(f"[INFO] Loading Trained PPO Policy...")
    model = PPO.load(model_path, env=env)

    print("[INFO] Starting Evaluation Loop...")
    obs = env.reset()
    
    raw_env = env.venv.envs[0].unwrapped
    dt = raw_env.model.opt.timestep * 10 
    
    while True:
        # start_time = time.time() 
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        time.sleep(0.05)

        # compute_time = time.time() - start_time
        # if compute_time < dt:
        #     time.sleep(dt - compute_time)

if __name__ == "__main__":
    main()