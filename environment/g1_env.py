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
        
        # set the properties containing the number of dimensions of the observation space and the actions space
            # NOTE: the observation space consists of:
                # 1 element indicating current desired behaviour (TODO: convert this primitive command vector into a proper command vector)
                    # 0 indicates the desired behaviour is standing still
                    # 1 indicates the desired behaviour is walking
                # 33 elements indicating the angular positions of every hinge joint (relative to the specific joint's parent bodies)
                    # NOTE: the first 4 elements are the quaternion indicating the orientation of the pelvis
                # 35 elements indicate the angular velocities of every hinge joint
                    # NOTE: the first 3 elements are the linear velocities of the pelvis relative to the world frame
                    # NOTE: the next 3 elements are the angular velocities of the pelvis about the x axis, y axis, and z axis of its own body frame
                    # NOTE: the other elements are the angular velocities of each joint in order of the joint definitions
            # NOTE: the action space has an element corresponding to each actuator on the robot (so for unitree g1 29)
        self.num_obs = 69
        self.num_actions = 29 

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

        # this line extracts the joint positions from the keyframe and stores them as the nominal_qpos
            # NOTE: when setting your nominal pose, it is super important that the feet of the robot start very close to the floor and start flat because if the feet penetrate the floor as a result of a policy action, oscillations causing jitter can occur where the policy actions lag the restoring forces exerted by the floor
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
        boundary_scaling_factor = 0.9

        # define bounds for the action scaling range such that the position targets are never set to a position that is outside the range of the value in this vector away from the nominal position of the joint
        npos_delta_lower = np.abs(self.joint_lims[:,0] - self.nominal_qpos[7:]) * boundary_scaling_factor
        npos_delta_upper = np.abs(self.joint_lims[:,1] - self.nominal_qpos[7:]) * boundary_scaling_factor 
        self.npos_upper = self.nominal_qpos[7:] + npos_delta_upper
        self.npos_lower = self.nominal_qpos[7:] - npos_delta_lower

        # save a value indicating the number of actons that have occured in the current episode of the current environment
        self.step_count = 0

        # initialize property values for the desired behaviour of the robot, the action number in the episode in which the desired behaviour switched from standing to walking, and a counter to hold how long the robot has been successfully standing for
            # for the desired behaviour property, 0 indicates the desired behaviour is standing still and 1 indicates the desired behaviour is walking
        self.desired_behaviour = 0
        self.des_bhve_switch_action_num = None
        self.stable_stand_count = 0

        # initializing the previous action to be the nominal position of the joints so that the action difference penalty is 0 at the first step and the learning policy does not get penalized for its first action being very different from the nominal pose
        self.previous_action = self.nominal_qpos[7:]

        # set up the properties needed to launch the viewer if desired
        self.render_mode = render_mode
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        else:
            self.viewer = None

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
                
        # define a vector to contain the factors by which corrections will be applied to each actuator
        correction_scaling = np.array([0.15, 0, 0.05, 0.15, 0.2, 0.01,
                                       0.15, 0, 0.05, 0.15, 0.2, 0.01,
                                       0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0])
        
        # initialize reward value
        total_reward = 0.0

        # set number of timesteps per action
        num_timesteps = 25

        # set the default action values based on the action vector that was sampled
            # scale the action outputted by the policy for the remained of the actuators (which is clipped by the spaces.Box line above) to surround the nominal position of each of the joints within a prespecified range (self.npos_delta)
        scaled_action = self.nominal_qpos[7:] + action * (self.npos_upper - self.npos_lower) / (self.box_high - self.box_low) * correction_scaling
        
        # if the desired behaviour is walking:
        if self.desired_behaviour == 1:

            # --- RANDOMIZED PUSH LOGIC ---
            # Clear external forces from the previous step
            self.data.xfrc_applied[self.pelvis_id, :] = 0.0
            
            # 0.5% chance to push per action (~once every 200 actions / 10 seconds)
            if self.np_random.uniform() < 0.005:
                # Curriculum scale: Starts at 10N, maxes out at 50N at 1,000,000 steps
                progress = min(1.0, self.total_steps / 10_000.0) # TODO: dividing by 1 million here is wrong, we usually only do 10000 timesteps per environment
                current_max_force = 10.0 + (40.0 * progress) 
                
                force_x = self.np_random.uniform(-current_max_force, current_max_force)
                force_y = self.np_random.uniform(-current_max_force, current_max_force)
                
                # Apply to Torso (Indices 0, 1 are Fx, Fy)
                self.data.xfrc_applied[self.pelvis_id, 0] = force_x
                self.data.xfrc_applied[self.pelvis_id, 1] = force_y

            # adding a spike push on the pelvis at the initial time which decays overtime 
            if self.step_count <= self.des_bhve_switch_action_num + 160:
                self.data.xfrc_applied[self.pelvis_id, 0] += 20 * np.sin(2*np.pi * ((self.step_count - self.des_bhve_switch_action_num)/320))

            # apply a upwards force that originally cancels out the weight of the robot but over time gradually transfers the weight to the robot
            # self.data.xfrc_applied[self.pelvis_id, 1] = 0.0
            # self.data.xfrc_applied[self.pelvis_id, 2] = 0.0
            # self.data.xfrc_applied[self.pelvis_id, 4] = 0.0
            # self.data.xfrc_applied[self.pelvis_id, 1] = -50 * self.data.qpos[1] * (1 - (self.total_steps / cutoff_timestep))
            # self.data.xfrc_applied[self.pelvis_id, 2] = 50 * (1 - (self.total_steps / cutoff_timestep))
            self.data.xfrc_applied[self.pelvis_id, 4] = -50 * np.arccos(np.dot(np.array([1,0,0,0]), self.data.qpos[3:7])) # * (1 - (self.total_steps / cutoff_timestep))

            # apply a positive force in the x direction to force the robot to move forward and maintain it's balance
            self.data.xfrc_applied[self.pelvis_id, 0] += 20
            
            scaled_action[0] -= self.nominal_qpos[7] # left hip pitch joint
            scaled_action[3] -= self.nominal_qpos[10] # left knee pitch joint
            scaled_action[6] -= self.nominal_qpos[13] # right hip pitch joint
            scaled_action[9] -= self.nominal_qpos[16] # right knee pitch joint


        # --- PHYSICS LOOP ---
        # NOTE: the number set here in this loop in combination with the timestep length set in the scene.xml file determines the control frequency of the robot
            # control frequency = 1 / (length of timestep * number of timesteps per action)
            # NOTE: the higher the control frequency, the more poses the robot can exist in that are maybe not optimal but feasible for it to maintain because it can issue commands so fast
        for phys_timestep in range(num_timesteps):
            
            # initialize a variable to hold the new action
                # this is necessary because during the walking gait, the new position is += to whats stored in this variable
            new_scaled_action = scaled_action.copy()

            # initialize variables to hold the behvaiour dependent penalties
            p_forward = 0
            p_stand_jitter = 0
            p_target_pose = 0

            # if the desired behaviour is standing still, calculate the jitter penalty
            if self.desired_behaviour == 0:

                # calculate the penalty for motion when the robot is supposed to be standing still
                p_stand_jitter = stand_jitter_penalty(self.data.qvel[6:], self.npos_lower, self.npos_upper)
                
            # if the desired behaviour is walking, calculate the limb positions
            elif self.desired_behaviour == 1:

                # initialize a variable to hold the current time
                curr_time = (self.step_count - self.des_bhve_switch_action_num) + phys_timestep/num_timesteps

                # determine what the target positions of the robot should be at the current timestep for the pitch motors of the thighs and knees for a walking gait
                left_thigh_target = np.deg2rad(15*np.cos(2*np.pi * ((curr_time)/40)) - 10)
                right_thigh_target = np.deg2rad(15*np.cos(2*np.pi * ((curr_time - 20)/40)) - 10)
                left_knee_target = np.deg2rad(25 * max(0, np.sin(2*np.pi * ((curr_time - 5)/40))) + 5)
                right_knee_target = np.deg2rad(25 * max(0, np.sin(2*np.pi * ((curr_time - 20 - 5)/40))) + 5)

                # add the target positions for the thigh and knee pitch joints for a walking gait to the corresponding elements of the scaled action vector so that the final target position for these joints is a combination of the target position for a walking gait and the correction outputted by the policy
                new_scaled_action[0] += left_thigh_target
                new_scaled_action[3] += left_knee_target
                new_scaled_action[6] += right_thigh_target
                new_scaled_action[9] += right_knee_target

                # calculate the penalty for the velocity of the robot not being the desired velocity
                p_forward = velocity_tracking_penalty(self.data.qvel[0])

                # calculate the penalty for the current position not being the target position
                p_target_pose = target_pose_deviation_penalty(self.data.qpos, curr_time, left_thigh_target, right_thigh_target, left_knee_target, right_knee_target)


            # # height of feet sites in xml files given by a distance sensor (measuring distance between the foot body and the ground geom)
            #     # NOTE: 0 when foot is on the ground, increases as foot gets higher off the ground (target estimate visual: 0.07)
            #     # NOTE: to see the feet sites in the viewer go to the rendering tab and toggle site 5 on
            # left_foot_height = self.data.sensor("left_foot_to_ground").data[0]
            # right_foot_height = self.data.sensor("right_foot_to_ground").data[0]
            
            # # contact force from touch sensors on the feet
            #     # NOTE: ~ 160 when foot is on the ground, decreases as foot gets higher off the ground (values around 30 when fallen over backwards, values around 20 when fallen over forwards)
            # left_foot_force = self.data.sensor("left_foot_touch").data[0]
            # right_foot_force = self.data.sensor("right_foot_touch").data[0]

            # calculate reward terms
            r_alive = alive_reward(self.data.qpos[2])
            # # calculate foot lift reward proportional to height of foot above floor
            # r_foot_lift = foot_lift_reward(left_foot_height, right_foot_height)
            # # calculate foot target penalty for keeping feet on the ground or lifting them too high
            # r_foot_target = foot_target_penalty(left_foot_height, right_foot_height)
            # calculate a reward depending on if the feet sites are contacting the ground at all
            # r_foot_contact = foot_contact_reward(left_foot_force, right_foot_force)
            
            # calculate penatly terms
            p_zvel = pelvis_zvel_penalty(self.data.qvel[2])
            p_pelvis_orientation = pelvis_orientation_penalty(self.data.qpos[3:7], self.data.qvel[3:6])

            # reward term weights
            w_alive = 1
            w_stand_jitter = 0.4
            w_velocity = 0.06
            w_zvel = 0.1
            w_pelvis_orientation = 0.1
            w_target_pose = 0.25

            # add to reward
            total_reward += w_alive*r_alive + w_stand_jitter*p_stand_jitter + w_zvel*p_zvel + w_pelvis_orientation*p_pelvis_orientation + w_velocity*p_forward + w_target_pose*p_target_pose
            
            # provide target angular positions to the PD controllers in the xml file by writing to mj.ctrl
            self.data.ctrl[:] = new_scaled_action

            # Step physics
            mujoco.mj_step(self.model, self.data)
        

        # check to see if the desired behaviour should be switched from standing still to walking
            # NOTE: this will eventually get replaced by user input setting the command vector
            # NOTE: this check needs to be put here at the bottom of the physics loop so that self.step_count can be used as the number of actions that were executed and resulted in a state where the robot was still standing
        if self.desired_behaviour == 0:

            # check if the robot is standing still currently
            standing_still_bool = standing_still(self.data.qvel)

            # if the robot is currently standing still:
            if standing_still_bool:

                # if the robot has been standing still for 1 second straight
                if self.stable_stand_count == 20:

                    # switch the desired behaviour from standing still to walking
                    self.desired_behaviour = 1

                    # save the time (action number) at which the desired behaviour switch occured
                        # TODO: check if this is right
                    self.des_bhve_switch_action_num = self.step_count

                # if the robot has not yet been standing still for 1 second straight
                else:

                    # increment the counter for the number of already completed consecutive actions the robot has stood still for
                        # NOTE: here by already completed actions we mean that because this check is at the end of the step function, this number is
                        # the number of actions that have been executed and after physics steps forward in time the robot is still standing still
                    self.stable_stand_count += 1

            # if the robot is not currently standing still
            else:

                # reset the counter indicating how many already completed consecutive actions the robot has stood still for
                self.stable_stand_count = 0


        # increment the episode length counter
        self.step_count += 1

        # store the current action as the previous action for the next step so that we can calculate the action difference penalty in the next step
        self.previous_action = scaled_action

        # collect a new observation to be supplied to the learning policy (and everything else) at the beginning of the computation stage of the next time
        obs = self._get_obs()

        # ================== Episode Termination Code ===========================
        
        # record the z height of the pelvis
            # NOTE: the pelvis is the freejoint of the unitree robot and it's z height is saved at index 2 of the qpos vector 
        pelvis_z = self.data.qpos[2] 
        
        # check the termination conditions to see if the current epsiode needs to be terminated
        if self.render_mode == "human":
            
            # sync the new state to the viewer
            self.viewer.sync()
            
            # Playback mode: Never reset, let it run infinitely
            terminated = False
            truncated = False
        else:
            
            # Training mode: Reset on fall or at 1000 steps
            terminated = bool(pelvis_z < 0.5) + bool(pelvis_z > 1)
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

        # reset the desired behaviour indicator, the counter indicating how many consecutive actions the robot has stood still for, and the time at which the desired behaviour switched from standing to walking
        self.desired_behaviour = 0
        self.stable_stand_count = 0
        self.des_bhve_switch_action_num = 0
        
        # based on the new set position and velocity values, calculate and populate all of the other values stored in mjModel and mjData
        mujoco.mj_forward(self.model, self.data)

        # we then allow the physics engine to step forward a few times to let the robot settle into the new pose after reset before we start returning observations and rewards to the agent. 
        # This is important because right after reset the robot might be in an unstable state and we dont want to penalize the agent for that or return observations that are not representative of the state it will actually be in when it starts taking actions.
        for _ in range(25):
                        
            mujoco.mj_step(self.model, self.data)
            
        return self._get_obs(), {}
    

    def _get_obs(self):
        
        # initialize a variable to hold the value indicating the desired behavior
        desired_behaviour = np.array([self.desired_behaviour])

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
        
        return np.concatenate([desired_behaviour, qpos, qvel]).astype(np.float32)


# ======================================================= State Assessment =========================================================

def standing_still(qvel):

    # pos_thresholds = np.array([0.1,
    #                            0.05, 0.05, 0.05, 0.05,
    #                            0.1, 0.05, 0.05, 0.1, 0.01, 0.01, 
    #                            0.1, 0.05, 0.05, 0.1, 0.01, 0.01, 
    #                            0.05, 0.05, 0.05, 
    #                            0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01,
    #                            0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01])
    
    vel_thresholds = np.array([0.1, 
                               0.1, 0.1, 0.1, 
                               0.05, 0.02, 0.02, 0.05, 0.01, 0.01, 
                               0.05, 0.02, 0.02, 0.05, 0.01, 0.01, 
                               0.1, 0.1, 0.1,
                               0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01,
                               0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01])
    
    # extracting from the nominal_qpos, qpos and qvel the relavent positions and velocities for deterimening whether the robot is standing in the crouch position
        # NOTE: this includes the z position of the pelvis, the angular orientation of the pelvis, the angular position of all the actuators, the angular velocity of all the actuators
        # NOTE: the x and y position of the pelvis will not be used as if an initial bounce or disturbance occurs but the robot remains in the crouch position at a different position that's still considered standing for our purposes
    # curr_pos_subset = qpos[2:36]
    # nominal_pos_subset = nominal_qpos[2:36]    
    curr_vel_subset = qvel[2:35]

    # defining a vector to contain the velocity corresponding to a perfectly still stand in the nominal pose (which is a vector of all 0 velocities)
    nominal_vel_subset = np.zeros_like(curr_vel_subset)

    # # calculating the absolute error between the curr position and the nominal position
    # pos_error = np.abs(curr_pos_subset - nominal_pos_subset)
    # print(pos_error)

    # # since there are two equivalent quaternions corresponding to each position (where one is an exact negation of the other) it's possible the physics engine outputs a sign flipped quaternion and so we must take the negative of that quaternion if it is negative for the position error
    #     # NOTE: if this is not the case the original position error from the line above is accurate
    #     # NOTE: curr_pos_subset[1:5] corresponds to the quaternion but we only use index 1 for the check as all of them will be negated if the second quaternion corresponding to a position is outputted by the physics engine
    # if curr_pos_subset[1] < 0:

    #     pos_error[1:5] = np.abs(-curr_pos_subset[1:5] - nominal_pos_subset[1:5])
    
    # calculating the absolute error between the curr velocities and the nominal velocity (vector of 0s)
    vel_error = np.abs(curr_vel_subset - nominal_vel_subset)

    # check if the velocity errors are outside the range of errors we consider to still be standing
    # pos_within_bounds = np.all(pos_error <= pos_thresholds)
    vel_within_bounds = np.all(vel_error <= vel_thresholds)

    return vel_within_bounds

# ======================================================= Rewards =========================================================


def alive_reward(pelvis_height):

    if 0.72 <= pelvis_height <= 0.80:
        return 1.0
    
    else:
        return - abs(pelvis_height - 0.76)
     
# def forward_motion_reward(forward_velocity):

#     return 1.0 * (forward_velocity > 0)

def stand_jitter_penalty(qvel, lower_bounds, upper_bounds):

    # initialize a boolean array indicating if each joint is frozen
    frozen_joints = ((upper_bounds - lower_bounds) == 0)

    # initialize an array to hold the range of motion through which each joint can rotate
        # NOTE: this is different from the joint limitations because the joint limitations are the maximum amount the limbs ever are allowed to move and this is the maximum amount the should move
        # TODO: fix this comment
    joint_pos_ranges = frozen_joints + (upper_bounds - lower_bounds)

    # initialize an array to hold correction factors to weight velocities in certain joints more than others
        # NOTE: these values were chosen based on importance of joints as well as how low their corresponding joint's angular velocity should be relative to the other joints (percentage of their range they should be going through per second)
    vel_corrections = np.ones(29)
    vel_corrections[0] = 3 # left hip pitch
    vel_corrections[6] = 3 # right hip pitch
    vel_corrections[3] = 2 # left knee pitch
    vel_corrections[9] = 2 # right knee pitch
    vel_corrections[4] = 4 # left ankle pitch
    vel_corrections[10] = 4 # right ankle pitch

    # normalize all the joint velocities relative to their max possible ranges of motion
    norm_qvel = ((qvel / joint_pos_ranges) * vel_corrections) / np.sum(~frozen_joints)

    # penalize the robot for how high the velocities of it's joints are
        # NOTE: this penalty should only be introduced in the balancing state to reduce flailing (we don't want joint velocity to be penalized when walking)
    return - np.sum(abs(norm_qvel))

def velocity_tracking_penalty(forward_velocity, target_velocity=1.0):

    velocity_error = abs(forward_velocity - target_velocity)
    return -velocity_error

def pelvis_zvel_penalty(pelvis_zvel):

    return - abs(pelvis_zvel)

# def foot_contact_reward(left_foot_force, right_foot_force, contact_threshold=50.0):

#     # reward for having one foot not in contact with the ground but not both feet off the ground
#         # NOTE: we assume a foot is off the ground if it's contact force is below a threshold
#     foot_contact_reward = ((left_foot_force<contact_threshold) + (right_foot_force<contact_threshold))%2

#     return foot_contact_reward

def pelvis_orientation_penalty(pelvis_orientation, pelvis_angular_velocity, target_orientation = np.array([1, 0, 0, 0])):

    # penalize the policy for having pelvis orientation close to perfectly upright
    pelvis_orientation_penalty = - np.sum(np.abs(pelvis_orientation - target_orientation))

    pelvis_angular_velocity_penalty = - np.sum(pelvis_angular_velocity)/3

    return pelvis_orientation_penalty + pelvis_angular_velocity_penalty# def target_pose_deviation_penalty(qpos, target_qpos, total_steps):

#     # calculate the deviation of the current pose from the nominal pose
#     pose_deviation = np.sum(np.abs(qpos - target_qpos))

#     return -pose_deviation

def target_pose_deviation_penalty(qpos, curr_time, left_thigh_target, right_thigh_target, left_knee_target, right_knee_target):

    # pelvis position target equation logic
        # because we switched to the force replacement logic and because of the way the axis are defined, the force acts in the direction we want the object to move (displacement between timesteps) a quarter period later since we modelled the system as a pendulum with a restoring force
        # this means that the force is phase shifted by a quarter period with the velocity (the velocity lags the force by a quarter period (v_phase_shift = -tau/4) assuming sinusoid equation in sin(2pi*(x+phase_shift)) )
        # however, because we know from the motion of a pendulum that the velocity is phase shifted with the position by another quarter period
        # so, the force is phase shifted a half period with the position (x_phase_shift = -tau/2)
        # in other words, the force acts in the direction opposite to the direction of the displacement from equilibrium (not displacement between timesteps)
        # and, the pelvis position equation was derived in this way from the force equations 
    pelvis_target = 0.06 * (-np.cos((2*np.pi) * ((curr_time))/40))

    return - ((abs(qpos[1] - pelvis_target)/0.12) + (abs(qpos[7] - left_thigh_target)/30) + (abs(qpos[13] - right_thigh_target)/30) + (abs(qpos[10] - left_knee_target)/50) + (abs(qpos[16] - right_knee_target)/50))


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