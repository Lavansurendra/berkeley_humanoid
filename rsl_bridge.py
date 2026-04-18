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
        
        self.num_envs = 1
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
        """Processes raw MuJoCo data into the grouped dictionary format."""
        # 1. Critic View (Truth)
        priv_t = torch.as_tensor(raw_truth, device=self.device).float().unsqueeze(0)

        # 2. Actor View (Partial + Noise)
        actor_obs_clean = raw_truth[2:] 
        noise = np.random.normal(0, self.noise_std, size=actor_obs_clean.shape)
        actor_obs_noisy = (actor_obs_clean + noise).astype(np.float32)
        actor_t = torch.as_tensor(actor_obs_noisy, device=self.device).float().unsqueeze(0)
        
        # We store them with names matching BerkeleyCfg.obs_groups
        new_obs = ObsDict()
        new_obs["obs"] = actor_t
        new_obs["privileged_obs"] = priv_t
        return new_obs

    def get_observations(self):
        return self.obs_dict

    def reset(self):
        raw_truth, _ = self.env.reset()
        self.obs_dict = self._process_raw_truth(raw_truth)
        # Modular rsl_rl expects the full dict from reset
        return self.obs_dict

    def step(self, actions):
            # 1. Convert action tensor [1, 12] to numpy [12]
            actions_np = actions.detach().cpu().numpy().squeeze(0)
            
            # 2. Step the MuJoCo physics engine
            raw_truth, reward, term, trunc, _ = self.env.step(actions_np)
            
            # 3. Process the new state into our ObsDict
            self.obs_dict = self._process_raw_truth(raw_truth)
            
            # 4. Prepare reward and done as flat [1] tensors
            rew_t = torch.tensor([reward], device=self.device, dtype=torch.float)
            # Done is True if the robot falls (term) OR the clock runs out (trunc)
            done_t = torch.tensor([term or trunc], device=self.device, dtype=torch.bool)
            
            # --- THE TIMEOUT FIX ---
            # 5. Tell the runner exactly *why* the episode ended so it can bootstrap
            infos = {
                "time_outs": torch.tensor([trunc], device=self.device, dtype=torch.bool)
            }
            
            # Return infos instead of {}
            return self.obs_dict, rew_t, done_t, infos