import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from environment.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

class HumanoidCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="berkeley", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq # Now interpreted as total timesteps
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        # Trigger based on total timesteps across all envs
        if (self.num_timesteps - self.last_time_trigger) >= self.save_freq:
            self.last_time_trigger = self.num_timesteps
            
            step_count = self.num_timesteps
            path_zip = os.path.join(self.save_path, f"{self.name_prefix}_{step_count}_steps")
            path_stats = os.path.join(self.save_path, f"{self.name_prefix}_{step_count}_stats.pkl")
            
            self.model.save(path_zip)
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path_stats)
            
            if self.verbose > 0:
                print(f"[CHECKPOINT] Saved at step {step_count}")
        return True

def main():
    xml_path = "environment/berkeley_scene.xml"
    num_envs = 4
    
    print("[INFO] Creating Parallel MuJoCo Environments...")
    

    env = make_vec_env(
        env_id=BerkeleyHumanoidMujocoEnv, 
        n_envs=num_envs,
        env_kwargs={"xml_path": xml_path, "render_mode": None},
        vec_env_cls=SubprocVecEnv
    )

    env = VecNormalize(
        env,
        norm_obs=True,     
        norm_reward=True,  
        clip_obs=10.0      
    )

    print("[INFO] Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        device="cpu"
    )

    # 1. Define your paths
    # Using 'output' as the parent directory we discussed
    models_dir = "output/models"
    checkpoints_dir = "output/checkpoints"

    # 2. Ensure the folders exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 3. Update the Callback
    checkpoint_callback = HumanoidCheckpointCallback(
        save_freq=10000, 
        save_path=checkpoints_dir, # Points to output/checkpoints
        name_prefix='berkeley_humanoid'
    )

    print("[INFO] Starting Training...")
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)

    print("[INFO] Saving final model...")
    # 4. Save to the models folder
    final_model_path = os.path.join(models_dir, "berkeley_humanoid_final_100000")
    stats_path = os.path.join(models_dir, "berkeley_humanoid_vecnormalize_100000.pkl")

    model.save(final_model_path)
    env.save(stats_path)

    print(f"[INFO] Training complete. Files saved in {models_dir}")

if __name__ == "__main__":
    main()