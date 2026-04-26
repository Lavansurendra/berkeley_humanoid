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

        # Get the ID for the torso body
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')

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
        # --- CURRICULUM TRACKER ---
        # Track total global steps across all resets to scale the push difficulty
        if not hasattr(self, 'total_steps'):
            self.total_steps = 0
            # Ensure torso ID is grabbed the first time we run a step
            self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.total_steps += 1
        
        # --- RANDOMIZED PUSH LOGIC ---
        # 1. Clear external forces from the previous step
        self.data.xfrc_applied[self.torso_id, :] = 0.0
        
        # 2. 0.5% chance to push per action (~once every 200 steps / 4 seconds)
        if self.np_random.uniform() < 0.005:
            # Curriculum scale: Starts at 30N, maxes out at 100N at 1,000,000 steps
            progress = min(1.0, self.total_steps / 1_000_000.0)
            current_max_force = 30.0 + (70.0 * progress) 
            
            force_x = self.np_random.uniform(-current_max_force, current_max_force)
            force_y = self.np_random.uniform(-current_max_force, current_max_force)
            
            # Apply to Torso (Indices 0, 1 are Fx, Fy)
            self.data.xfrc_applied[self.torso_id, 0] = force_x
            self.data.xfrc_applied[self.torso_id, 1] = force_y

        # --- ACTION APPLICATION ---
        # Scale AI output to +/- 0.3 radians around the crouch
        target_q = action * 0.3 + self.nominal_qpos
        
        # initialize reward value
        total_reward = 0.0

        # set number of timesteps per action
        num_timesteps = 25

        # --- PHYSICS LOOP ---
        # NOTE: the number set here in this loop in combination with the timestep length set in the berkeley_scene.xml file determines the control frequency of the robot
            # control frequency = 1 / (length of timestep * number of timesteps per action)
            # NOTE: the higher the control frequency, the more poses the robot can exist in that are maybe not optimal but feasible for it to maintain because it can issue commands so fast
        for _ in range(num_timesteps):
            
            # calculate reward terms
            r_alive = alive_reward()

            # reward term weights
            w_alive = 0.1

            # add to reward
            total_reward += w_alive*r_alive

            # 1. CRITICAL: Clear all joint forces before calculating new ones
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

        # TODO: fix this so the rendering speed is independent from the control frequency
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        
        # # Base Reward
        # reward = 1.0
        
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
        
        return obs, total_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Set Torso Position and Orientation
        # Adjust 0.65 to the height where the feet just touch the ground
        self.data.qpos[0:3] = [0, 0, 0.55]  # [x, y, z]
        self.data.qpos[3:7] = [1, 0, 0, 0] # Unit quaternion (upright)
        
        # 2. Set Joint Positions (Crouch)
        # We add a tiny bit of noise so the robot doesn't start in a "mathematically perfect" trap
        noise = self.np_random.uniform(low=-0.01, high=0.01, size=len(self.nominal_qpos))
        self.data.qpos[7:] = self.nominal_qpos + noise
        
        # 3. Clear velocities (optional noise here too)
        self.data.qvel[:] = self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        
        self.step_count = 0
        
        # 4. Settle the Physics
        # mj_forward just calculates positions. 
        # mj_step(self.model, self.data) actually runs the solver.
        # Running 5-10 steps with zero actions lets the robot "land" properly.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
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

def alive_reward():
    return 1.0