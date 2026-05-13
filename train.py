import os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from environment.berkeley_env import BerkeleyEnv
from environment.g1_env import G1Env

class HumanoidCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="g1", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq # TODO: model weights should be saved once every few epochs, not based on a set number of timesteps
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
    xml_path = "environment/g1_scene.xml" # filepath in the current directory specifying where the .xml file is that defines the world used in every environment
    num_envs = 20 # the number of environments that will be running in parallel (set this to the number of threads (logical cores) on your computer)
    total_timesteps_per_env = 20000 # the number of timesteps per environment per training cycle
    total_timesteps = total_timesteps_per_env * num_envs
    
    # Store hyperparams in a dict for W&B tracking
    config = {
        "policy_type": "MlpPolicy", # keyword determining what type of neural network architecture you are using, in this case MlpPolicy specifies you are using 2 MLP networks
        
        "total_timesteps": total_timesteps, # this is the total number of timesteps per training cycle (NOT THE NUMBER OF TIMESTEPS TOTAL PER ENVIRONMENT)
            # this can be verified by noting that this element of the config dictionary is supplied to the model.learn function call
            # which then supplies this number to the super().learn function call (which calls the .learn method of the OnPolicyAlgorithm class which is a parent class to the PPO class)
            # which then checks to make sure the total number of timesteps that have passed so far is less than this number
            # NOTE: this number is NOT the total number of timesteps per environment, instead it is just the total number of timesteps that will occur in a singel training cycle where 
            # every time all the environments are stepped once in a single iteration of the rollout while loop, 1 timestep times the number of environments running in parallel is added to the counter
            # that is checked to be less than this variable to keep training going
        "learning_rate": 3e-4,
        "n_steps": 2048, # this is the number of timesteps (NOT TIMES!) per environment per rollout
            # this can be verified by noting that this element of the config dictionary is supplied to the PPO function call
            # which then initializes an instance of the PPO class found in the stable_baselines3/ppo/ppo.py file alongside all the attributes and methods of its parent class (the OnPolicyAlgorithm class found in the stable_baselines3/common/on_policy_algorithm.py file).
            # this means that when the model.learn function is called down below, the .learn method of the PPO class is called which then calls the .learn method of the OnPolicyAlgorithm class
            # which then supplies to the collect_rollouts method of the OnPolicyAlgorithm class self.n_steps (.n_steps is the attribute of the OnPolicyAlgorithm class containing this value)
            # which then has a while loop containing all the function calls needed to make everything that happens in each timestep of a rollout happen
            # where one of the things that happens in each iteration is every environment is stepped once and the n_steps variable (from the collect_rollouts function not here) is incremented by 1
            # such that every iteration the condition being checked by the while loop is if n_steps is less than n_rollout_steps which now contains the value stored in this variable 
            # NOTE: the total number of timesteps per rollout can be found by multiplying this number by the number of environments
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
        env_id=G1Env, 
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
        save_freq=100000, 
        save_path=checkpoints_dir,
        name_prefix='g1'
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
    final_model_path = os.path.join(models_dir, f"g1_final_{total_timesteps}")
    stats_path = os.path.join(models_dir, f"g1_vecnormalize_{total_timesteps}.pkl")

    model.save(final_model_path)
    env.save(stats_path)

    # Close the W&B run
    run.finish()
    print(f"[INFO] Training complete. Files saved in {models_dir}")

if __name__ == "__main__":
    main()