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

        # 2. STIFFNESS (Kp) and DAMPING (Kd)
        # Mapped from: HXX(10, 1.5), HFE(15, 1.5), KFE(15, 1.5), FFE(1, 0.1), FAA(1, 0.1)
        self.kp = np.array([
            10.0, 10.0, 15.0, 15.0, 1.0, 1.0,  # Left Leg
            10.0, 10.0, 15.0, 15.0, 1.0, 1.0   # Right Leg
        ])
        self.kd = np.array([
            1.5, 1.5, 1.5, 1.5, 0.1, 0.1,  # Left Leg
            1.5, 1.5, 1.5, 1.5, 0.1, 0.1   # Right Leg
        ])

        # Motor Saturation Limits (from config.json)
        # Order: HR, HAA, HFE, KFE, FFE, FAA
        self.effort_limits = np.array([
            20.0, 20.0, 30.0, 30.0, 20.0, 5.0,  # Left Leg
            20.0, 20.0, 30.0, 30.0, 20.0, 5.0   # Right Leg
        ], dtype=np.float32)

        self.velocity_limits = np.array([
            23, 23, 20, 14, 20, 42,  # Left Leg:  HR, HAA, HFE, KFE, FFE, FAA
            23, 23, 20, 14, 23, 42   # Right Leg: HR, HAA, HFE, KFE, FFE, FAA
        ])

        # Nominal Stance (The Crouch)
        self.nominal_qpos = np.array([
            -0.071,   # LL_HR
            0.103,   # LL_HAA
            -0.463,   # LL_HFE
            0.983,   # LL_KFE
            -0.350,   # LL_FFE
            0.126,   # LL_FAA
            0.071,   # LR_HR
            -0.103,   # LR_HAA
            -0.463,   # LR_HFE
            0.983,   # LR_KFE
            -0.350,   # LR_FFE
            -0.126    # LR_FAA
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
            
            # 4. CLAMP TORQUE: Apply exact hardware limits per joint
            tau = np.clip(tau, -self.effort_limits, self.effort_limits)
            
            # 5. Apply only to the 12 leg joints
            self.data.qfrc_applied[6:] = tau
            
            # 6. Step physics
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1

        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        # NOTE: a feet_slide_reward is at the bottom of this file, it came from the original Isaac Lab code and is translated to work with MuJoCo's API. You can call it here and add it to the reward if you want to penalize foot sliding.
        reward = 1.0
        
        # Termination: End if torso falls below 0.3m
        torso_z = self.data.qpos[2] 
        
        if self.render_mode == "human":
            # Playback mode: Never reset, let it run infinitely
            terminated = False
            truncated = False
        else:
            # Training mode: Reset on fall or at 1000 steps
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
    
def feet_slide_reward(model, data, foot_body_ids, action):

    sliding_penalty = 0.0
    
    for foot_id in foot_body_ids:
        # Check if foot is in contact with the floor
        # MuJoCo handles contacts differently; we iterate through the contact array
        in_contact = False
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == foot_id or contact.geom2 == foot_id:
                in_contact = True
                break
                
        if in_contact:
            # Get linear velocity of the foot
            # data.cvel gives 6D spatial velocity (3 rot, 3 lin) for each body
            foot_vel = data.cvel[foot_id][3:5] # X and Y velocity
            vel_norm = np.linalg.norm(foot_vel)
            
            # Penalize the magnitude of velocity while in contact
            sliding_penalty += vel_norm
            
    return -sliding_penalty # Negative because it's a penalty