import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class BerkeleyHumanoidMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        
        # 1. Load the model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 2. Save a master copy of the motor strengths for randomization
        self.original_gear = self.model.actuator_gear.copy()
        
        # 3. Setup tracking and IDs
        self.total_steps = 0
        self.step_count = 0
        self.num_actions = 12 
        # Observation space: 19 (qpos) + 18 (qvel) = 37 total
        self.num_obs = self.model.nq + self.model.nv 
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')

        # 4. Joint Gains (Berkeley standard Kp/Kd)
        self.kp = np.array([
            10.0, 10.0, 15.0, 15.0, 1.0, 1.0,  # Left Leg
            10.0, 10.0, 15.0, 15.0, 1.0, 1.0   # Right Leg
        ])
        self.kd = np.array([
            1.5, 1.5, 1.5, 1.5, 0.1, 0.1,  # Left Leg
            1.5, 1.5, 1.5, 1.5, 0.1, 0.1   # Right Leg
        ])

        # 5. Hardware Torque Limits (Nm)
        self.effort_limits = np.array([
            20.0, 20.0, 30.0, 30.0, 20.0, 5.0,  # Left Leg
            20.0, 20.0, 30.0, 30.0, 20.0, 5.0   # Right Leg
        ], dtype=np.float32)

        # 6. Optimized Crouch Pose (qpos[7:])
        # Order: Yaw, Roll, Pitch, Knee, AnkleP, AnkleR
        self.nominal_qpos = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0, # Left Leg
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0  # Right Leg
        ])

        # 7. Gym Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        self.render_mode = render_mode
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

    def step(self, action):
        self.total_steps += 1
        self.step_count += 1
        
        # --- EXTERNAL DISTURBANCE (PUSH) ---
        self.data.xfrc_applied[self.torso_id, :] = 0.0
        if self.np_random.uniform() < 0.005:
            progress = min(1.0, self.total_steps / 1_000_000.0)
            current_max_force = 30.0 + (70.0 * progress) 
            force_x = self.np_random.uniform(-current_max_force, current_max_force)
            force_y = self.np_random.uniform(-current_max_force, current_max_force)
            self.data.xfrc_applied[self.torso_id, 0] = force_x
            self.data.xfrc_applied[self.torso_id, 1] = force_y

        # --- PD CONTROL ---
        target_q = action * 0.3 + self.nominal_qpos
        total_reward = 0.0
        num_timesteps = 50

        for _ in range(num_timesteps):
            # Calculate simple alive reward
            total_reward += (0.1 * alive_reward())

            # Apply PD Control to leg joints (Index 6+ in qfrc/qvel)
            self.data.qfrc_applied[:] = 0.0
            current_q = self.data.qpos[7:]
            current_v = self.data.qvel[6:]
            
            tau = self.kp * (target_q - current_q) - self.kd * current_v
            tau = np.clip(tau, -self.effort_limits, self.effort_limits)
            
            # Map torque to leg joints
            self.data.qfrc_applied[6:] = tau
            mujoco.mj_step(self.model, self.data)
        
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        # --- CHECK STATUS ---
        obs = self._get_obs()
        torso_z = self.data.qpos[2] 
        
        if self.render_mode == "human":
            terminated = False
            truncated = False
        else:
            # Robot falls if torso height < 0.3m
            terminated = bool(torso_z < 0.3)
            truncated = self.step_count >= 1000 
        
        return obs, total_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # --- DOMAIN RANDOMIZATION ---
        # Motor strength jitter (+/- 5%)
        self.model.actuator_gear[:, 0] = self.original_gear[:, 0] * self.np_random.uniform(0.95, 1.05)

        # Joint damping jitter (simulates mechanical wear)
        self.model.dof_damping[:] = self.np_random.uniform(0.5, 3.0)

        # Surface friction jitter
        self.model.geom_friction[:, 0] = self.np_random.uniform(0.5, 1.2)

        # Ground softness jitter (0.02=Hard, 0.4=Soft)
        for i in range(self.model.ngeom):
            if "floor" in self.model.geom(i).name or "plane" in self.model.geom(i).name:
                self.model.geom_solref[i] = [self.np_random.uniform(0.02, 0.4), 1.0]

        # --- POSITION RESET ---
        # Torso height and upright orientation
        self.data.qpos[0:3] = [0, 0, 0.55] 
        self.data.qpos[3:7] = [1, 0, 0, 0] 
        
        # Joint crouch with tiny noise
        noise = self.np_random.uniform(low=-0.01, high=0.01, size=len(self.nominal_qpos))
        self.data.qpos[7:] = self.nominal_qpos + noise
        
        # Zero out velocities
        self.data.qvel[:] = self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.step_count = 0
        
        # Let physics settle so robot doesn't start with a "jerk"
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flatten()
        ]).astype(np.float32)

# --- REWARDS ---

def alive_reward():
    return 1.0