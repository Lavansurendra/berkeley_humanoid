import os
import time
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from exts.berkeley_humanoid.berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

def main():
    # Path to your converted MJCF/XML scene
    xml_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml"
    
    # Path to the trained model and the normalization stats
    model_path = "berkeley_humanoid_final_1000000.zip"
    stats_path = "vec_normalize.pkl" # This is the file saved by train.py
    
    print(f"[INFO] Loading MuJoCo Environment...")
    
    # 1. WRAP IN DUMMY VEC ENV
    # SB3 requires the base environment to be in a vectorized container first
    base_env_fn = lambda: BerkeleyHumanoidMujocoEnv(xml_path=xml_path, render_mode="human")
    env = DummyVecEnv([base_env_fn])
    
    # 2. LOAD AND FREEZE NORMALIZATION STATS
    if os.path.exists(stats_path):
        print("[INFO] Loading normalization statistics...")
        env = VecNormalize.load(stats_path, env)
        
        # --- CRITICAL FIX FOR PLAYBACK ---
        env.training = False      # Stop updating the observation averages
        env.norm_reward = False   # Do not normalize rewards during visual evaluation
    else:
        print(f"[WARNING] Could not find {stats_path}. The model will likely flail wildly!")

    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Have you run train.py yet?")
        return

    print(f"[INFO] Loading Trained PPO Policy...")
    model = PPO.load(model_path, env=env)

    print("[INFO] Starting Evaluation Loop...")
    # 3. VEC ENV RESET FIX
    # DummyVecEnv returns only `obs`, not `obs, info`
    obs = env.reset()
    
    # Run the simulation indefinitely so you can watch the robot
    while True:
        start_time = time.time() 
        
        # The model predicts the best action based on the observation
        action, _states = model.predict(obs, deterministic=True)
        
        # 4. VEC ENV STEP FIX
        # DummyVecEnv returns 4 values (term/trunc are combined into `done`)
        obs, reward, done, info = env.step(action)
        
        # Note: DummyVecEnv automatically resets the environment when done is True,
        # so you don't need to manually call env.reset() here anymore.

        # 5. ACCESSING THE BASE ENV FOR THE SPEED FIX
        # Because we are wrapped twice (DummyVecEnv -> VecNormalize -> BaseEnv), 
        # we have to dig down to get the raw MuJoCo timestep.
        dt = env.envs[0].unwrapped.model.opt.timestep * 10 
        compute_time = time.time() - start_time
        
        # If the math finished faster than 0.02s, pause the loop to lock it to real-time
        if compute_time < dt:
            time.sleep(dt - compute_time)

if __name__ == "__main__":
    main()