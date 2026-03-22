import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
# from berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv
from exts.berkeley_humanoid.berkeley_humanoid.tasks.locomotion.velocity.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

def main():
    # Path to your converted MJCF/URDF file
    # You need to generate an MJCF that includes the robot and a floor
    xml_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_scene.xml"
    
    # Create parallel environments (Number of CPU cores you want to use)
    num_envs = 8
    
    print("[INFO] Creating MuJoCo Vectorized Environments...")
    env = make_vec_env(
        lambda: BerkeleyHumanoidMujocoEnv(xml_path=xml_path, render_mode=None), 
        n_envs=num_envs
    )

    # Initialize PPO Agent 
    # Hyperparameters translated roughly from agents/rsl_rl_cfg.py
    print("[INFO] Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        learning_rate=1e-3,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        device="cpu"
        # tensorboard_log="./logs/berkeley_ppo_tensorboard/"
    )

    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path='./logs/checkpoints/',
        name_prefix='berkeley_humanoid'
    )

    print("[INFO] Starting Training...")
    # 30,000 iterations * 24 steps (from your rsl_rl_cfg.py) = ~720,000 total steps
    model.learn(total_timesteps=10_000, callback=checkpoint_callback)
    
    print("[INFO] Saving final model...")
    model.save("berkeley_humanoid_final_10000")

if __name__ == "__main__":
    main()