import gymnasium as gym
import numpy as np
# This imports the class we just spent time editing
from berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

# 1. Start the environment
# IMPORTANT: Make sure your .xml file name matches exactly here
# Use ./ to tell MuJoCo "Look right here in the current folder"
env = BerkeleyHumanoidMujocoEnv(xml_path="./berkeley_scene.xml", render_mode="human")

print("Robot is starting... watch the MuJoCo window!")

try:
    while True:
        # 2. Reset the robot to the crouch position
        obs, info = env.reset()
        
        # 3. Let it sit for 200 steps (about 4 seconds)
        for _ in range(200):
            # We send zeros (no movement) to test if the crouch is stable
            obs, reward, terminated, truncated, info = env.step(np.zeros(12))
            
        print("Resetting to test domain randomization...")

except KeyboardInterrupt:
    print("\nStopped by user.")
    env.close()