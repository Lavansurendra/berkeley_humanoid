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
        
        # 22 DOF + Base state (Position, Orientation, Linear Vel, Angular Vel)
        # Adjust these based on whether you are doing 12 DOF (legs only) or 22 DOF
        self.num_actions = 22 
        self.num_obs = 47 # Example: Base (3+3), Joints (22), Vels (19)

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

        # Tracking variables for rewards
        self.step_count = 0
        self.last_action = np.zeros(self.num_actions)

    def step(self, action):
        # Apply PD control (Simplified for translation)
        # In Isaac Lab, this was handled by `actuator_pd.py`
        target_positions = action # Scale this by your joint limits
        self.data.ctrl[:] = target_positions
        
        # Advance physics
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self.step_count >= 1000 # Max episode length
        
        self.last_action = action
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        
        # Reset to default joint positions (from your berkeley_humanoid.py init_state)
        # self.data.qpos[...] = default_positions
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Extract qpos (positions) and qvel (velocities)
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten()
        ]).astype(np.float32)

    def _compute_reward(self, action):
        # Placeholder for the complex reward logic
        # You will integrate the translated rewards.py functions here
        survival_reward = 1.0
        return survival_reward

    def _check_terminated(self):
        # End episode if the torso falls below a certain height
        torso_z = self.data.qpos[2] 
        return bool(torso_z < 0.4)