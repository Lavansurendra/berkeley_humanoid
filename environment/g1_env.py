import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, render_mode=None):
        
        '''

        qpos vector structure (in order of elements):
        ------------
        - pelvis freejoint (elements 0-6): 
            - 0: pelvis x coordinate relative to world frame (meters)
            - 1: pelvis y coordinate relative to world frame (meters)
            - 2: pelvis z coordinate relative to world frame (meters)
            - 3: pelvis orientation quaternion element 1 (cos(a/2) where a is in radians)
            - 4: pelvis orientation quaternion element 2 (Ux*sin(a/2) where a is in radians)
            - 5: pelvis orientation quaternion element 3 (Uy*sin(a/2) where a is in radians)
            - 6: pelvis orientation quaternion element 4 (Uz*sin(a/2) where a is in radians)

        - left leg hinge joints (elements 7-12):
            - 7: left hip pitch joint (radians)
            - 8: left hip roll joint (radians)
            - 9: left hip yaw joint (radians)
            - 10: left knee pitch joint (radians)
            - 11: left ankle pitch joint (radians)
            - 12: left ankle roll joint (radians)

        - right leg hinge joints (elements 13-18):
            - 13: right hip pitch joint (radians)
            - 14: right hip roll joint (radians)
            - 15: right hip yaw joint (radians)
            - 16: right knee pitch joint (radians)
            - 17: right ankle pitch joint (radians)
            - 18: right ankle roll joint (radians)

        - waist hinge joints (elements 19-21):
            - 19: waist yaw joint (radians)
            - 20: waist roll joint (radians)
            - 21: waist pitch joint (radians)

        - left arm hinge joints (elements 22-28):
            - 22: left shoulder pitch joint (radians)
            - 23: left shoulder roll joint (radians)
            - 24: left shoulder yaw joint (radians)
            - 25: left elbow pitch joint (radians)
            - 26: left wrist roll joint (radians)
            - 27: left wrist pitch joint (radians)
            - 28: left wrist yaw joint (radians)

        - right arm hinge joints (elements 28-35):
            - 29: right shoulder pitch joint (radians)
            - 30: right shoulder roll joint (radians)
            - 31: right shoulder yaw joint (radians)
            - 32: right elbow pitch joint (radians)
            - 33: right wrist roll joint (radians)
            - 34: right wrist pitch joint (radians)
            - 35: right wrist yaw joint (radians)
        
        
        '''
        
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_actions = 29 
        self.num_obs = 68

        # in the xml for the keyframe named "crouch" the robot is in a crouching position which we will use as our nominal pose to scale our actions around
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "crouch")
        
        # this line extracts the joint positions from the keyframe and stores them as the nominal_qpos.
        # NOTE: we slice [7:] to skip the x,y,z positions and quaternion of the floating base
        self.nominal_qpos = self.model.key_qpos[key_id][7:]

        
        # retrieving the ID for the pelvis body (used for applying random pushes in the step function)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        # This variable will track the total number of steps taken across all episodes, which we can use to scale the difficulty of the random pushes over time (curriculum learning)
        self.total_steps = 0

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
        '''
        Inputs
        ------
        action:
        - 
        
                

        '''
        
        # we will use the total_steps variable to track how many steps the agent has taken across all episodes, and use that to scale the difficulty of the random pushes over time. 
        # this way, the agent starts with easier conditions and gradually faces more challenging perturbations as it learns.
        self.total_steps += 1
        
        # --- RANDOMIZED PUSH LOGIC ---
        # 1. Clear external forces from the previous step
        self.data.xfrc_applied[self.pelvis_id, :] = 0.0
        
        # 2. 0.5% chance to push per action (~once every 200 steps / 4 seconds)
        if self.np_random.uniform() < 0.005:
            # Curriculum scale: Starts at 30N, maxes out at 100N at 1,000,000 steps
            progress = min(1.0, self.total_steps / 1_000_000.0)
            current_max_force = 10.0 + (40.0 * progress) 
            
            force_x = self.np_random.uniform(-current_max_force, current_max_force)
            force_y = self.np_random.uniform(-current_max_force, current_max_force)
            
            # Apply to Torso (Indices 0, 1 are Fx, Fy)
            self.data.xfrc_applied[self.pelvis_id, 0] = force_x
            self.data.xfrc_applied[self.pelvis_id, 1] = force_y

        # --- ACTION APPLICATION ---
        # Scale AI output to +/- 0.3 radians around the crouch
        target_q = action * 0.3 + self.nominal_qpos
        
        # initialize reward value
        total_reward = 0.0

        # set number of timesteps per action
        num_timesteps = 25

        # # define the timestep duration (dt) in seconds based on the timestep length defined in the scene.xml file and the number of timesteps we want to run for each action
        # dt = self.model.opt.timestep * num_timesteps

        # # set the maximum allowed change in angle per timestep based on velocity limits and timestep duration (dt)
        # max_delta_q = self.velocity_limits * dt

        # --- PHYSICS LOOP ---
        # NOTE: the number set here in this loop in combination with the timestep length set in the scene.xml file determines the control frequency of the robot
            # control frequency = 1 / (length of timestep * number of timesteps per action)
            # NOTE: the higher the control frequency, the more poses the robot can exist in that are maybe not optimal but feasible for it to maintain because it can issue commands so fast
        for _ in range(num_timesteps):
            
            # calculate reward terms
            r_alive = alive_reward()
            r_forward = forward_motion_reward(self.data.qvel[0])
            
            # calculate penatly terms
            px_velocity = velocity_tracking_reward(self.data.qvel[0]) # x velocity of the pelvis is at index 0 of the qvel vector

            # reward term weights
            w_alive = 0.1
            w_velocity = 0.9

            # add to reward
            total_reward += w_alive*r_alive + w_velocity*px_velocity + w_velocity*r_forward           
            
            # provide target angular positions to the PD controllers in the xml file by writing to mj.ctrl
            self.data.ctrl[:] = target_q

            # 6. Step physics
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1

        # TODO: fix this so the rendering speed is independent from the control frequency
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
        
        # # Base Reward
        # reward = 1.0
        
        # record the z height of the pelvis
            # NOTE: the pelvis is the freejoint of the unitree robot and it's z height is saved at index 2 of the qpos vector 
        pelvis_z = self.data.qpos[2] 
        
        if self.render_mode == "human":
            # Playback mode: Never reset, let it run infinitely
            terminated = False
            truncated = False
        else:
            # Training mode: Reset on fall or at 1000 steps
            terminated = bool(pelvis_z < 0.3)
            truncated = self.step_count >= 1000 
        
        return obs, total_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # specifically resetting to the crouching keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "crouch")
        self.data.qpos[:] = self.model.key_qpos[key_id]

        # when we reset to the crouching keyframe we must also reset the control array to match the crouching pose
        # so that the PD controllers in the xml file dont apply huge forces trying to get the robot to the target pose defined by the control array which would cause it to explode on reset
        self.data.ctrl[:] = self.model.key_qpos[key_id][7:]

        # adding a little noise to each of the joints
        # (We skip 0:7 to avoid the torso's x,y,z and quaternion)
        noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.num_actions)
        self.data.qpos[7:] += noise
        
        # 3. Clear velocities (optional noise here too)
        self.data.qvel[:] = self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        
        # reset the step count for the episode so that curriculum learning can start again with the smallest random pushes
        self.step_count = 0
        
        # now using mj_forward (which is a command that calculates positions and velocities based on the current state of the robot
        # we can update the qpos and qvel vectors to reflect the changes we made to the robot's state in the reset function (like setting it to the crouching keyframe and adding noise) before we start stepping the physics forward in time.
            # NOTE: unlike with mj_step, mj_forward does not run the numerical solver and cause the robot to move it simply updates the position and velocity information based on the current state of the robot.
        mujoco.mj_forward(self.model, self.data)

        # we then allow the physics engine to step forward a few times to let the robot settle into the new pose after reset before we start returning observations and rewards to the agent. 
        # This is important because right after reset the robot might be in an unstable state and we dont want to penalize the agent for that or return observations that are not representative of the state it will actually be in when it starts taking actions.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        return self._get_obs(), {}
    

    def _get_obs(self):
        
        # we define the qpos vector without the x, y and z position of the pelvis as we don't assume that we can calculated that information
        # we include all orientations of the pelvis as we assume that the robot can sense its orientation using an IMU or similar sensor, and we include all joint positions as we assume the robot can sense those using joint encoders or similar sensors
        # [3:7] = Pelvis Orientation (Quaternion)
        # [7:36] = 29 Joint Positions
        qpos = self.data.qpos[3:36] 

        # --- QVEL (35 values) ---
        # we keep the full qvel vector which includes the linear and angular velocity of the pelvis as well as the joint velocities because we assume that the robot can sense all of that information using an IMU and joint encoders or similar sensors.
        # [0:3] = Pelvis Linear Velocity
        # [3:6] = Pelvis Angular Velocity
        # [6:35] = 29 Joint Velocities
        qvel = self.data.qvel[0:35]
        
        return np.concatenate([qpos, qvel]).astype(np.float32)
    

# ======================================================= Rewards =========================================================

def alive_reward():
    return 1.0

def forward_motion_reward(forward_velocity):

    return forward_velocity


def velocity_tracking_reward(forward_velocity, target_velocity=1.0):

    velocity_error = abs(forward_velocity - target_velocity)
    return -velocity_error

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