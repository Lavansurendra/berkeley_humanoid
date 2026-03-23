import os
import time
import mujoco
from stable_baselines3 import PPO
from exts.berkeley_humanoid.berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

def main():
    # Path to your converted MJCF/XML scene
    xml_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml"
    
    # Path to the trained model saved by train.py
    model_path = "berkeley_humanoid_final_1000000.zip"
    
    print(f"[INFO] Loading MuJoCo Environment...")
    # Initialize the environment with human render mode so you can see it
    env = BerkeleyHumanoidMujocoEnv(xml_path=xml_path, render_mode="human")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Have you run train.py yet?")
        return

    print(f"[INFO] Loading Trained PPO Policy...")
    model = PPO.load(model_path, env=env)

    print("[INFO] Starting Evaluation Loop...")
    obs, info = env.reset()
    
    # Run the simulation indefinitely so you can watch the robot
    while True:
        # 2. START THE STOPWATCH
        start_time = time.time() 
        
        # The model predicts the best action based on the observation
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment forward using the chosen action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # if terminated or truncated:
        #      obs, info = env.reset()

        # 3. APPLY THE SPEED FIX
        # Grab the physics timestep (0.002s) and multiply by our 10x decimation (0.02s)
        dt = env.unwrapped.model.opt.timestep * 10 
        compute_time = time.time() - start_time
        
        # If the math finished faster than 0.02s, pause the loop to lock it to real-time
        if compute_time < dt:
            time.sleep(dt - compute_time)

if __name__ == "__main__":
    main()