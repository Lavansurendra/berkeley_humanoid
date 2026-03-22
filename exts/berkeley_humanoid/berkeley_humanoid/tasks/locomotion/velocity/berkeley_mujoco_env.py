import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class BerkeleyHumanoidMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_actions = 12 
        self.num_obs = self.model.nq + self.model.nv 

        # --- REAL-WORLD ACCURACY TUNING ---
        # The main Berkeley Humanoid is stiff. We use 30.0 as a 'High-Fidelity' baseline.
        # We increase Kd (damping) to 1.5 to 'soak up' the numerical noise and prevent NaNs.
        self.kp = 30.0
        self.kd = 2.0

        # Nominal Stance (The Crouch)
        self.nominal_qpos = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # Left Leg
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0   # Right Leg
        ])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.render_mode = render_mode
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.step_count = 0

    def step(self, action):
        # Scale AI output to +/- 0.3 radians around the crouch
        target_q = action * 0.3 + self.nominal_qpos
        
        for _ in range(10):
            # 1. CRITICAL: Clear all forces before calculating new ones
            self.data.qfrc_applied[:] = 0.0
            
            # 2. Get current state (Indices 7+ for pos, 6+ for vel)
            current_q = self.data.qpos[7:]
            current_v = self.data.qvel[6:]
            
            # 3. PD Formula: τ = Kp(target - current) - Kd(velocity)
            tau = self.kp * (target_q - current_q) - self.kd * current_v
            
            # 4. CLAMP TORQUE: Real motors have a limit (approx 40Nm for this robot)
            # This prevents the 'Infinite Force' explosion (NaNs)
            tau = np.clip(tau, -40.0, 40.0)
            
            # 5. Apply only to the 12 leg joints
            self.data.qfrc_applied[6:] = tau
            
            # 6. Step physics
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1

        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        reward = 1.0 # To be replaced by the Reward Team
        
        # Termination: End if torso falls below 0.3m
        torso_z = self.data.qpos[2] 
        terminated = bool(torso_z < 0.3)
        truncated = self.step_count >= 1000 
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Set to Crouch
        self.data.qpos[7:] = self.nominal_qpos
        self.data.qvel[:] = 0.0
        
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten()
        ]).astype(np.float32)