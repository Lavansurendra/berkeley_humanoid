import numpy as np
import traceback

# Import your custom environment directly
from exts.berkeley_humanoid.berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

def run_dummy_check():
    print("========================================")
    print("   BERKELEY HUMANOID MUJOCO TEST")
    print("========================================\n")
    
    # Using the updated path you set
    xml_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml"
    
    try:
        print("[INFO] 1. Loading Environment...")
        # We use render_mode=None for a fast, headless check
        env = BerkeleyHumanoidMujocoEnv(xml_path=xml_path, render_mode=None)
        
        print(f"[SUCCESS] Environment loaded!")
        print(f"  -> Observation Space: {env.observation_space.shape}")
        print(f"  -> Action Space: {env.action_space.shape}\n")
        
        print("[INFO] 2. Testing env.reset()...")
        obs, info = env.reset()
        print(f"[SUCCESS] Reset successful! Initial obs shape: {obs.shape}\n")
        
        print("[INFO] 3. Testing 10 random steps...")
        for i in range(10):
            # Sample a random action that perfectly matches your action_space
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
        print("[SUCCESS] Successfully took 10 random steps without crashing!\n")
        print("========================================")
        print(" ALL SYSTEMS GO. YOU ARE READY TO TRAIN.")
        print("========================================")
        
    except Exception as e:
        print("\n[FAILED] The dummy check encountered an error:")
        print("--------------------------------------------------")
        traceback.print_exc()
        print("--------------------------------------------------")
        print("Look at the error above. It is likely a dimension mismatch between your URDF and the env.py file.")

if __name__ == "__main__":
    run_dummy_check()