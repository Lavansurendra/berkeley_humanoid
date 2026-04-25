import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from environment.berkeley_mujoco_env import BerkeleyHumanoidMujocoEnv

class HumanoidCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="berkeley", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq 
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.save_freq:
            self.last_time_trigger = self.num_timesteps
            
            step_count = self.num_timesteps
            path_zip = os.path.join(self.save_path, f"{self.name_prefix}_{step_count}_steps")
            path_stats = os.path.join(self.save_path, f"{self.name_prefix}_{step_count}_stats.pkl")
            
            self.model.save(path_zip)
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path_stats)
            
            if self.verbose > 0:
                print(f"[CHECKPOINT] Saved model and stats at step {step_count}")
        return True

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :return: (function)
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func

def main():
    # --- CONFIGURATION ---
    xml_path = "environment/berkeley_scene.xml"
    num_envs = 4
    total_timesteps = 1_000_000
    
    # Store hyperparams in a dict for W&B tracking
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "num_envs": num_envs,
    }

    # 1. Initialize Weights & Biases
    # This will prompt you to log in the first time you run it locally.
    run = wandb.init(
        project="wiscohumanoids",
        config=config,
        sync_tensorboard=True,  # This tells W&B to mirror your TensorBoard logs
        monitor_gym=True,       # Auto-upload videos if you use a RecordVideo wrapper
        save_code=True,         # Keeps a snapshot of this script on the cloud
    )

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
    # Add tensorboard_log path so SB3 knows where to save the local data
    model = PPO(
        config["policy_type"], 
        env, 
        n_steps=config["n_steps"],
        learning_rate=linear_schedule(config["learning_rate"]),
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        clip_range=0.2,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log=f"output/logs/{run.id}", 
        device="cpu"
    )

    # Prepare directories
    models_dir = "output/models"
    checkpoints_dir = "output/checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 2. Setup Callbacks
    # Custom checkpointing for Weights + Normalization Stats
    checkpoint_callback = HumanoidCheckpointCallback(
        save_freq=10000, 
        save_path=checkpoints_dir,
        name_prefix='berkeley_humanoid'
    )

    # W&B callback for system metrics (psutil) and gradient tracking
    wandb_callback = WandbCallback(
        model_save_path=f"output/models/{run.id}",
        verbose=2,
    )

    # Combine them into a single list
    callback_list = CallbackList([checkpoint_callback, wandb_callback])

    print("[INFO] Starting Training...")
    model.learn(
        total_timesteps=config["total_timesteps"], 
        callback=callback_list
    )

    print("[INFO] Saving final model...")
    final_model_path = os.path.join(models_dir, f"berkeley_humanoid_final_{total_timesteps}")
    stats_path = os.path.join(models_dir, f"berkeley_humanoid_vecnormalize_{total_timesteps}.pkl")

    model.save(final_model_path)
    env.save(stats_path)

    # Close the W&B run
    run.finish()
    print(f"[INFO] Training complete. Files saved in {models_dir}")

if __name__ == "__main__":
    main()