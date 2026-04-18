import torch
import numpy as np

# A simple dictionary that supports .to(device) for legacy runners
class ObsDict(dict):
    def to(self, device):
        for k, v in self.items():
            if hasattr(v, "to"):
                self[k] = v.to(device)
        return self

class RSLRL_Bridge:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.num_envs = self.env.num_envs
        self.num_obs = 35          
        self.num_privileged_obs = 37 
        self.num_actions = 12
        self.noise_std = 0.05       

        # Metadata for the runner's logger
        self.cfg = {
            "num_envs": self.num_envs,
            "env_name": "berkeley_humanoid_mujoco_v2",
            "device": self.device
        }

        self.obs_dict = ObsDict()

    def _process_raw_truth(self, raw_truth):
        """Processes batched MuJoCo data into the grouped dictionary format."""
        # 1. Critic View (Truth) 
        # Removed .unsqueeze(0) because raw_truth already has a batch dimension [8, 37]
        priv_t = torch.as_tensor(raw_truth, device=self.device).float()

        # 2. Actor View (Partial + Noise)
        # Added ":" to drop the first two X/Y coordinates across ALL 8 environments
        actor_obs_clean = raw_truth[:, 2:] 
        noise = np.random.normal(0, self.noise_std, size=actor_obs_clean.shape)
        actor_obs_noisy = (actor_obs_clean + noise).astype(np.float32)
        
        # Removed .unsqueeze(0) here as well
        actor_t = torch.as_tensor(actor_obs_noisy, device=self.device).float()
        
        # We store them with names matching BerkeleyCfg.obs_groups
        new_obs = ObsDict()
        new_obs["obs"] = actor_t
        new_obs["privileged_obs"] = priv_t
        return new_obs

    def get_observations(self):
        return self.obs_dict

    def reset(self):
        # SB3's VecEnv returns ONLY the observation array [8, 37], not a tuple!
        raw_truth = self.env.reset()
        self.obs_dict = self._process_raw_truth(raw_truth)
        # Modular rsl_rl expects the full dict from reset
        return self.obs_dict

    def step(self, actions):
        # 1. Convert batched action tensor [8, 12] to numpy array (8, 12)
        # Removed .squeeze(0) so the batch dimension remains intact
        actions_np = actions.detach().cpu().numpy()
        
        # 2. Step the 8 parallel MuJoCo physics engines
        # SB3 VecEnv returns exactly 4 items (obs, rewards, dones, infos)
        raw_truth, rewards, dones, infos = self.env.step(actions_np)
        
        # 3. Process the batched state into our ObsDict
        self.obs_dict = self._process_raw_truth(raw_truth)
        
        # 4. Prepare reward and done as batched [8] tensors
        rew_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_t = torch.tensor(dones, device=self.device, dtype=torch.bool)
        
        # --- THE TIMEOUT FIX FOR VEC ENV ---
        # 5. infos is a list of 8 dictionaries. We loop through them to find timeouts.
        timeouts = torch.tensor([
            info.get("TimeLimit.truncated", False) for info in infos
        ], device=self.device, dtype=torch.bool)
        
        extras = {"time_outs": timeouts}
        
        return self.obs_dict, rew_t, done_t, extras