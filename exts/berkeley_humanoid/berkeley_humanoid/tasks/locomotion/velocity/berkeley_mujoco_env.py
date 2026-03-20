import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class BerkeleyHumanoidMujocoEnv(gym.Env):
    """Custom Environment for Berkeley Humanoid using MuJoCo and Gymnasium."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        
        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # --- DYNAMIC SIZING FIX ---
        # We read the dimensions directly from the loaded robot
        # self.model.nv = total velocities (6 base + 12 joints = 18)
        # self.model.nq = total positions (7 base + 12 joints = 19)
        self.num_actions = 12 # Based on the 12 DOFs for the legs
        self.num_obs = self.model.nq + self.model.nv 

        # Action space: Target joint positions (normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        
        # Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.render_mode = render_mode
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.step_count = 0

    def step(self, action):
        target_positions = action
        
        # --- THE ACTUATOR FIX ---
        # Since the URDF has no <motor> tags, we apply force directly to the joints.
        # We skip the first 6 DOFs (which belong to the floating base) and apply to the 12 leg joints.
        num_base_dofs = self.model.nv - self.num_actions
        self.data.qfrc_applied[num_base_dofs:] = target_positions
        
        # Advance physics
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        reward = 1.0 # Placeholder
        
        # Check termination: End if torso z-height falls below 0.4 meters
        torso_z = self.data.qpos[2] 
        terminated = bool(torso_z < 0.4)
        truncated = self.step_count >= 1000 
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten()
        ]).astype(np.float32)