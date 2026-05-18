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
        
        # initialize variables to hold python struct objects containing all the static and dynamic information about the current state of the environment
            # NOTE: the following is my current guess but I don't understand this
            # a python struct object is an object composed of a set of numpy arrays where each numpy array corresponds to one member of a C struct
            # mujoco.MjModel creates an instance of the mjModel class (which is a python struct object set up to hold all the data about the current state of the environment)
            # the .from_xml_path() method then populates the new python struct object with all the data from the scene files
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_actions = 29 
        self.num_obs = 68

        # initialize an array to hold the lower and upper limits of the range of motion of each joint (excluding the freejoint) in radians relative to the joint reference points (right now set to 0 rad)
            # NOTE: the joint limits for the hip pitch joints were both artificially adjusted to (-1, 1) from their previous values of (-2.5307, 2.8798)
            # NOTE: the joint limits for the hip roll joints were both artificially adjusted to (-0.5, 0.1) from their previous values of (-2.9671 0.5236)
            # NOTE: the joint limits for both arms were fixed to the nominal position
            # NOTE: the joint limit for the hip yaw joints were both artificially adjusted to (-0.1, 0.1) from their previous values of (-2.7576, 2.7576)
            # NOTE: the joint limits for the waist yaw joint was artificially adjusted to (-0.3, 0.3) from it's previous values of (-2.618, 2.618)
            # NOTE: the joint limits for the waist roll joint was artificially adjusted to (-0.2, 0.2) from it's previous values of (-0.52, 0.52)
        self.joint_lims = np.array(
            [[-1, 1], [-0.5, 0.1], [-0.1, 0.1], [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
            [-1, 1], [-0.1, 0.5], [-0.1, 0.1], [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
            [-0.3, 0.3], [-0.2, 0.2], [-0.52, 0.52], 
            [0.2, 0.2], [0.2, 0.2], [0, 0], [1.28, 1.28], [0, 0], [0, 0], [0, 0], 
            [0.2, 0.2], [-0.2, -0.2], [0, 0], [1.28, 1.28], [0, 0], [0, 0], [0, 0]])


        # in the xml for the keyframe named "crouch" the robot is in a crouching position which we will use as our nominal pose to scale our actions around
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "crouch")
        # # in the xml for the keyframe named "step" the robot is in the initial step position
        # key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "step")
        # # in the xml for the keyframe named "stand" the robot is in a standing position
        # key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        # # in the xml for the keyframe named "half_step" the robot is in the middle of taking a step
        # key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "half_step")

        # this line extracts the joint positions from the keyframe and stores them as the nominal_qpos.
        # NOTE: we slice [7:] to skip the x,y,z positions and quaternion of the floating base
        self.nominal_qpos = self.model.key_qpos[key_id]
        
        # retrieving the ID for the pelvis body (used for applying random pushes in the step function)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        # This variable will track the total number of steps taken across all episodes, which we can use to scale the difficulty of the random pushes over time (curriculum learning)
        self.total_steps = 0

        # defining a lower and upper bound for the spaces.Box. We currently believe that the actions of the policy are limited so that they always fall within this range via clipping
            # TODO: determine if this is actually done by direct clipping or through the application of some function who's range is limited to this range such as tanh.
        self.box_low = -1.0
        self.box_high = 1.0

        # setting the bounds of the action and observation spaces
        self.action_space = spaces.Box(low=self.box_low, high=self.box_high, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        # define a scaling factor which will be used to determine the range of values surrounding the nominal pose that the action given by the policy (and clipped by the spaces.Box) will be scaled to.
        self.bound_scaling_factor = 0.9

        # define bounds for the action scaling range such that the position targets are never set to a position that is outside the range of the value in this vector away from the nominal position of the joint
        self.npos_delta_lower = np.abs(self.joint_lims[:,0] - self.nominal_qpos[7:]) * self.bound_scaling_factor
        self.npos_delta_upper = np.abs(self.joint_lims[:,1] - self.nominal_qpos[7:]) * self.bound_scaling_factor 

        self.npos_upper = self.nominal_qpos[7:] + self.npos_delta_upper
        self.npos_lower = self.nominal_qpos[7:] - self.npos_delta_lower

        self.render_mode = render_mode
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.step_count = 0

        # initializing the previous action to be the nominal position of the joints so that the action difference penalty is 0 at the first step and the learning policy does not get penalized for its first action being very different from the nominal pose
        self.previous_action = self.nominal_qpos[7:]

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
        
        # # --- RANDOMIZED PUSH LOGIC ---
        # # Clear external forces from the previous step
        # self.data.xfrc_applied[self.pelvis_id, :] = 0.0
        
        # # 0.5% chance to push per action (~once every 200 actions / 10 seconds)
        # if self.np_random.uniform() < 0.005:
        #     # Curriculum scale: Starts at 10N, maxes out at 50N at 1,000,000 steps
        #     progress = min(1.0, self.total_steps / 1_000_000.0) # TODO: dividing by 1 million here is wrong, we usually only do 10000 timesteps per environment
        #     current_max_force = 10.0 + (40.0 * progress) 
            
        #     force_x = self.np_random.uniform(-current_max_force, current_max_force)
        #     force_y = self.np_random.uniform(-current_max_force, current_max_force)
            
        #     # Apply to Torso (Indices 0, 1 are Fx, Fy)
        #     self.data.xfrc_applied[self.pelvis_id, 0] = force_x
        #     self.data.xfrc_applied[self.pelvis_id, 1] = force_y

        
        # # ------------ Cyclical Thigh Pushing Logic -----------------
        # # set the cutoff timestep
        #     # TODO: change this from being hardcoded to being a parameter
        cutoff_timestep = 10000
        
        # # clear the force applied on each thigh and the pelvis from the previous step
        # self.data.qfrc_applied[6] = 0.0 # left hip pitch joint
        # self.data.qfrc_applied[12] = 0.0 # right hip pitch joint
        # self.data.xfrc_applied[self.pelvis_id, 0] = 0.0
        # self.data.xfrc_applied[self.pelvis_id, 1] = 0.0

        # # apply a constant force on the pelvis that cuts off part way through the training
        # self.data.xfrc_applied[self.pelvis_id, 0] = 20 * (1 - (self.total_steps / cutoff_timestep))

        # # apply a cyclical force on the pelvis in the y direction that cuts off part way through the training (this gets the foot ground clearance to swing)
        #     # NOTE: the force needs to act in the opposite direction from the direction you want the pelvis to swing (when the left foot is swinging, the pelvis needs to move right so the force should act left)
        #     # NOTE: a positive force corresponds to a force left
        # # self.data.xfrc_applied[self.pelvis_id, 1] = 10 * (np.cos((2*np.pi) * (self.step_count/50)) * (1 - (self.total_steps / cutoff_timestep)))
        # self.data.xfrc_applied[self.pelvis_id, 1] = 10 * (np.cos((2*np.pi) * (self.step_count/80)) * (1 - (self.total_steps / cutoff_timestep)))

        # # calculate the angular position (pitch) of where the thighs should be at the current timestep for a walking gait
        # left_thigh_angle = 30*np.cos(2*np.pi*(self.step_count / 80))
        # right_thigh_angle = 30*np.cos(2*np.pi*((self.step_count - 40) / 80))

        # # calculate the force that should be applied at the current time on each thigh
        #     # NOTE: positive forces make the legs go backwards
        #     # NOTE: there is a slight time delay between the force being applied to the thighs that would cause them to swing and the force on the pelvis in the y direction and that allows there to be some ground clearnace before the thigh swing begins
        # left_thigh_qfrc = 80 * (-np.sin(np.pi * (left_thigh_angle/60)))
        # right_thigh_qfrc = 80 * (-np.sin(np.pi * (right_thigh_angle/60)))

        # # set the force being applied for the current time on each thigh
        # self.data.qfrc_applied[6] = left_thigh_qfrc * (1 - (self.total_steps / cutoff_timestep)) # left hip pitch joint
        # self.data.qfrc_applied[12] = right_thigh_qfrc * (1 - (self.total_steps / cutoff_timestep)) # right hip pitch joint

        # # ------------ Cyclical Knee Pushing Logic -----------------
        
        # # clear the force applied on each thigh and the pelvis from the previous step
        # self.data.qfrc_applied[9] = 0.0 # left knee pitch joint
        # self.data.qfrc_applied[15] = 0.0 # right knee pitch joint

        # # calculate the angular position (pitch) of where the thighs should be at the current timestep for a walking gait
        # left_knee_angle = 50*max(0, np.sin(2*np.pi*(self.step_count/80)))
        # right_knee_angle = 50*max(0, np.sin(2*np.pi*((self.step_count - 40)/80)))


        # # calculate the force that should be applied at the current time on each knee
        #     # NOTE: positive forces make the shin go backwards
        # # left_knee_qfrc = 40 * np.sin((2*np.pi) * ((self.total_steps - 5)/50))
        # # right_knee_qfrc = 40 * np.sin((2*np.pi) * ((self.total_steps - 30)/50))
        # left_knee_qfrc = 40 * (-np.sin(np.pi * ((left_knee_angle - 25)/50)))
        # right_knee_qfrc = 40 * (-np.sin(np.pi * ((right_knee_angle - 25)/50)))

        # # set the force being applied for the current time on each knee
        # self.data.qfrc_applied[9] = left_knee_qfrc * (1 - (self.total_steps / cutoff_timestep)) # left knee pitch joint
        # self.data.qfrc_applied[15] = right_knee_qfrc * (1 - (self.total_steps / cutoff_timestep)) # right knee pitch joint

        # define a vector to contain the factors by which corrections will be applied to each actuator
        correction_scaling = np.array([0.15, 0, 0.05, 0.15, 0.1, 0.01,
                                       0.15, 0, 0.05, 0.15, 0.1, 0.01,
                                       0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0])

        # scale the action outputted by the policy for the remained of the actuators(which is clipped by the spaces.Box line above) to surround the nominal position of each of the joints within a prespecified range (self.npos_delta)
        scaled_action = self.nominal_qpos[7:] + action * (self.npos_upper - self.npos_lower) / (self.box_high - self.box_low) * correction_scaling
        
        # adding a spike push on the pelvis at the initial time which decays overtime 
        if 40 <= self.step_count <= 60:
            self.data.xfrc_applied[self.pelvis_id, 0] = 0.0
            self.data.xfrc_applied[self.pelvis_id, 0] = 20 * np.sin(2*np.pi * ((self.step_count - 40)/40))

        elif self.step_count > 60:
            # apply a upwards force that originally cancels out the weight of the robot but over time gradually transfers the weight to the robot
            self.data.xfrc_applied[self.pelvis_id, 1] = 0.0
            self.data.xfrc_applied[self.pelvis_id, 2] = 0.0
            self.data.xfrc_applied[self.pelvis_id, 4] = 0.0
            # self.data.xfrc_applied[self.pelvis_id, 1] = -50 * self.data.qpos[1] * (1 - (self.total_steps / cutoff_timestep))
            # self.data.xfrc_applied[self.pelvis_id, 2] = 50 * (1 - (self.total_steps / cutoff_timestep))
            self.data.xfrc_applied[self.pelvis_id, 4] = -50 * np.arccos(np.dot(np.array([1,0,0,0]), self.data.qpos[3:7])) # * (1 - (self.total_steps / cutoff_timestep))

            # apply a positive force in the x direction to force the robot to move forward and maintain it's balance
            self.data.xfrc_applied[self.pelvis_id, 0] = 20
            
            # scaled_action[0] *= min((self.total_steps / cutoff_timestep),1) # left hip pitch joint
            # scaled_action[3] *= min((self.total_steps / cutoff_timestep),1) # left knee pitch joint
            # scaled_action[6] *= min((self.total_steps / cutoff_timestep),1) # right hip pitch joint
            # scaled_action[9] *= min((self.total_steps / cutoff_timestep),1) # right knee pitch joint
            scaled_action[0] -= self.nominal_qpos[7] # left hip pitch joint
            scaled_action[3] -= self.nominal_qpos[10] # left knee pitch joint
            scaled_action[6] -= self.nominal_qpos[13] # right hip pitch joint
            scaled_action[9] -= self.nominal_qpos[16] # right knee pitch joint

    
        # initialize reward value
        total_reward = 0.0

        # set number of timesteps per action
        num_timesteps = 25

        # --- PHYSICS LOOP ---
        # NOTE: the number set here in this loop in combination with the timestep length set in the scene.xml file determines the control frequency of the robot
            # control frequency = 1 / (length of timestep * number of timesteps per action)
            # NOTE: the higher the control frequency, the more poses the robot can exist in that are maybe not optimal but feasible for it to maintain because it can issue commands so fast
        for phys_timestep in range(num_timesteps):
            
            # initialize a variable to hodl the new action
            new_scaled_action = scaled_action.copy()

            if self.step_count > 40:

                # determine what the target positions of the robot should be at the current timestep for the pitch motors of the thighs and knees for a walking gait
                left_thigh_target = np.deg2rad(10*np.cos(2*np.pi * ((self.step_count + phys_timestep/num_timesteps - 60)/40)))
                right_thigh_target = np.deg2rad(10*np.cos(2*np.pi * ((self.step_count + phys_timestep/num_timesteps - 20 - 60)/40)))
                left_knee_target = np.deg2rad(25 * max(0, np.sin(2*np.pi * ((self.step_count + phys_timestep/num_timesteps - 60)/40))))
                right_knee_target = np.deg2rad(25 * max(0, np.sin(2*np.pi * ((self.step_count + phys_timestep/num_timesteps - 20 - 60)/40))))

                # add the target positions for the thigh and knee pitch joints for a walking gait to the corresponding elements of the scaled action vector so that the final target position for these joints is a combination of the target position for a walking gait and the correction outputted by the policy
                # new_scaled_action[0] += left_thigh_target * (1 - min((self.total_steps / cutoff_timestep), 1))
                # new_scaled_action[3] += left_knee_target * (1 - min((self.total_steps / cutoff_timestep), 1))
                # new_scaled_action[6] += right_thigh_target * (1 - min((self.total_steps / cutoff_timestep), 1))
                # new_scaled_action[9] += right_knee_target * (1 - min((self.total_steps / cutoff_timestep), 1))
                new_scaled_action[0] += left_thigh_target
                new_scaled_action[3] += left_knee_target
                new_scaled_action[6] += right_thigh_target
                new_scaled_action[9] += right_knee_target


            # # height of feet sites in xml files given by a distance sensor (measuring distance between the foot body and the ground geom)
            #     # NOTE: 0 when foot is on the ground, increases as foot gets higher off the ground (target estimate visual: 0.07)
            #     # NOTE: to see the feet sites in the viewer go to the rendering tab and toggle site 5 on
            # left_foot_height = self.data.sensor("left_foot_to_ground").data[0]
            # right_foot_height = self.data.sensor("right_foot_to_ground").data[0]
            
            # contact force from touch sensors on the feet
                # NOTE: ~ 160 when foot is on the ground, decreases as foot gets higher off the ground (values around 30 when fallen over backwards, values around 20 when fallen over forwards)
            left_foot_force = self.data.sensor("left_foot_touch").data[0]
            right_foot_force = self.data.sensor("right_foot_touch").data[0]

            # calculate reward terms
            r_alive = alive_reward(self.data.qpos[2])
            # r_forward = forward_motion_reward(self.data.qvel[0])
            # # calculate foot lift reward proportional to height of foot above floor
            # r_foot_lift = foot_lift_reward(left_foot_height, right_foot_height)
            # # calculate foot target penalty for keeping feet on the ground or lifting them too high
            # r_foot_target = foot_target_penalty(left_foot_height, right_foot_height)
            # calculate a reward depending on if the feet sites are contacting the ground at all
            # r_foot_contact = foot_contact_reward(left_foot_force, right_foot_force)
            
            
            # calculate penatly terms
            # p_limits = motor_limit_penalty(action, self.joint_lims)
            # p_action_diff = action_diff_penalty(scaled_action, self.previous_action)
            p_zvel = pelvis_zvel_penalty(self.data.qvel[2])
            p_pelvis_orientation = pelvis_orientation_penalty(self.data.qpos[3:7])
            # p_target_pose_deviation = target_pose_deviation_penalty(self.data.qpos, self.step_count, phys_timestep, num_timesteps)
            # px_velocity = velocity_tracking_reward(self.data.qvel[0]) # x velocity of the pelvis is at index 0 of the qvel vector

            # reward term weights
            w_alive = 1
            w_velocity = 0.06
            w_zvel = 0.1
            w_pelvis_orientation = 0.1
            # w_target_pose_deviation = 0.3

            # add to reward
            total_reward += w_alive*r_alive + w_zvel*p_zvel + w_pelvis_orientation*p_pelvis_orientation
            
            # provide target angular positions to the PD controllers in the xml file by writing to mj.ctrl
            self.data.ctrl[:] = new_scaled_action

            # 6. Step physics
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1

        # store the current action as the previous action for the next step so that we can calculate the action difference penalty in the next step
        self.previous_action = scaled_action

        # TODO: fix this so the rendering speed is independent from the control frequency
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        obs = self._get_obs()
    
        
        # record the z height of the pelvis
            # NOTE: the pelvis is the freejoint of the unitree robot and it's z height is saved at index 2 of the qpos vector 
        pelvis_z = self.data.qpos[2] 
        
        if self.render_mode == "human":
            # Playback mode: Never reset, let it run infinitely
            terminated = False
            truncated = False
        else:
            # Training mode: Reset on fall or at 1000 steps
            terminated = bool(pelvis_z < 0.2) + bool(pelvis_z > 1)
            truncated = self.step_count >= 1000 
        
        return obs, total_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # specifically resetting to the nominal pose
        self.data.qpos[:] = self.nominal_qpos

        # when we reset to the nominal pose we must also reset the control array to match the nominal pose
        # so that the PD controllers in the xml file dont apply huge forces trying to get the robot to the target pose defined by the control array which would cause it to explode on reset
        # NOTE: here because the control values necessarily must be the same as the nominal position values, we can reset the control values using teh nominal position values
        self.data.ctrl[:] = self.nominal_qpos[7:]

        # adding a little noise to each of the joints
        # (We skip 0:7 to avoid the torso's x,y,z and quaternion)
        noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.num_actions)
        self.data.qpos[7:] += noise
        
        # 3. Clear velocities (optional noise here too)
        self.data.qvel[:] = self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        
        # reset the step count for the episode so that curriculum learning can start again with the smallest random pushes
        self.step_count = 0
        
        # based on the new set position and velocity values, calculate and populate all of the other values stored in mjModel and mjData
        mujoco.mj_forward(self.model, self.data)

        # we then allow the physics engine to step forward a few times to let the robot settle into the new pose after reset before we start returning observations and rewards to the agent. 
        # This is important because right after reset the robot might be in an unstable state and we dont want to penalize the agent for that or return observations that are not representative of the state it will actually be in when it starts taking actions.
        for _ in range(25):
                        
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

# def motor_limit_penalty(action, joint_lims):

#     '''
#     Penalize the agent if the action is outside the joint limitations
#     The reason we are adding this reward is because we want the learning policy to have no limitations on the action values it can output
#     so that it can explore the full range of possible actions and find the best ones that maximize the reward however we also
#     need it to learn that it should only output values that are actually possible for the robot to execute hence the penalty
#     '''
    
#     # calculate the distance between the angular positions specified in the action vector and the lower and upper limits of the joint rangees of motion
#     lower_lim_dist = np.abs(np.minimum(action - joint_lims[:,0], np.zeros_like(action)))
#     upper_lim_dist = np.maximum(action - joint_lims[:,1], np.zeros_like(action))

#     # calculate an offset term so that as the actions get closer to being inside the limits the penalty does not go to 0 and there is a still a penalty for being outside the limits
#     offset = ((action <= joint_lims[:,0]) + (action >= joint_lims[:,1])).astype(int)

#     return -(np.sum(lower_lim_dist) + np.sum(upper_lim_dist)) - np.sum(offset)


def alive_reward(pelvis_height):

    if 0.72 <= pelvis_height <= 0.80:
        return 1.0
    
    else:
        return - abs(pelvis_height - 0.76)

     
# def action_diff_penalty(action, prev_action):

#     return -np.sum(np.abs(action-prev_action))

def forward_motion_reward(forward_velocity):

    return 1.0 * (forward_velocity > 0)


# def forward_motion_reward(forward_velocity):

#     return forward_velocity

def pelvis_zvel_penalty(pelvis_zvel):

    return - abs(pelvis_zvel)

def foot_contact_reward(left_foot_force, right_foot_force, contact_threshold=50.0):

    # reward for having one foot not in contact with the ground but not both feet off the ground
        # NOTE: we assume a foot is off the ground if it's contact force is below a threshold
    foot_contact_reward = ((left_foot_force<contact_threshold) + (right_foot_force<contact_threshold))%2

    return foot_contact_reward

def pelvis_orientation_penalty(pelvis_orientation, target_orientation = np.array([1, 0, 0, 0])):

    # penalize the policy for having pelvis orientation close to perfectly upright
    pelvis_orientation_penalty = - np.sum(np.abs(pelvis_orientation - target_orientation))

    return pelvis_orientation_penalty

# def target_pose_deviation_penalty(qpos, target_qpos, total_steps):

#     # calculate the deviation of the current pose from the nominal pose
#     pose_deviation = np.sum(np.abs(qpos - target_qpos))

#     return -pose_deviation

def target_pose_deviation_penalty(qpos, total_timestep, curr_physics_timestep, tot_num_phys_timesteps):

    # Pelvis position target equation logic
        # because we switched to the force replacement logic and because of the way the axis are defined, the force acts in the direction we want the object to move (displacement between timesteps) a quarter period later since we modelled the systme as a pendulum with a restoring force
        # this means that the force is phase shifted by a quarter period with the velocity (the velocity lags the force by a quarter period (v_phase_shift = -tau/4) assuming sinusoid equation in sin(2pi*(x+phase_shift)) )
        # however, because we know from the motion of a pendulum that the velocity is phase shifted with the position by another quarter period
        # so, the force is phase shifted a half period with the position (x_phase_shift = -tau/2)
        # in other words, the force acts in the direction opposite to the direction of the displacement from equilibrium (not displacement between timesteps)
    # 
    pelvis_target = 0.06 * (-np.cos((2*np.pi) * ((total_timestep + (curr_physics_timestep/tot_num_phys_timesteps))/160)))
    left_thigh_target = 30*np.cos(2*np.pi * ((total_timestep + (curr_physics_timestep/tot_num_phys_timesteps))/160))
    right_thigh_target = 30*np.cos(2*np.pi * ((total_timestep + (curr_physics_timestep/tot_num_phys_timesteps) - 80)/160))
    left_knee_target = 50 * max(0, np.sin(2*np.pi * ((total_timestep + (curr_physics_timestep/tot_num_phys_timesteps))/160)))
    right_knee_target = 50 * max(0, np.sin(2*np.pi * ((total_timestep + (curr_physics_timestep/tot_num_phys_timesteps) - 80)/160)))

    return - ((abs(qpos[1] - pelvis_target)/0.12) + (abs(qpos[7] - left_thigh_target)/60) + (abs(qpos[13] - right_thigh_target)/60) + (abs(qpos[10] - left_knee_target)/100) + (abs(qpos[16] - right_knee_target)/100))



def velocity_tracking_reward(forward_velocity, target_velocity=1.0):

    velocity_error = abs(forward_velocity - target_velocity)
    return -velocity_error

def foot_lift_reward(left_foot_height, right_foot_height, left_foot_force, right_foot_force, contact_threshold=50.0):
    
    # check if both feet off of ground
    left_foot_force_below_threshold = left_foot_force < contact_threshold
    right_foot_force_below_threshold = right_foot_force < contact_threshold
    both_feet_off_ground = (left_foot_force_below_threshold * right_foot_force_below_threshold)
    
    # reward for lifting feet off the ground (0 if both feet not supporting weight on ground)
    left_foot_reward = left_foot_height *(1-both_feet_off_ground)
    right_foot_reward = right_foot_height *(1-both_feet_off_ground)

    return left_foot_reward + right_foot_reward

def foot_target_penalty(left_foot_height, right_foot_height, left_foot_force, right_foot_force, target_height=0.07, contact_threshold=50.0):

    # check if either foot is off the ground
        # NOTE: we assume a foot is off the ground if it's contact force is above a threshold
    left_foot_force_below_threshold = left_foot_force < contact_threshold
    right_foot_force_below_threshold = right_foot_force < contact_threshold

    # penalty for distance from feet target height
    left_foot_reward = -abs(left_foot_height - target_height) * left_foot_force_below_threshold
    right_foot_reward = -abs(right_foot_height - target_height) * right_foot_force_below_threshold

    return left_foot_reward + right_foot_reward



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