import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class BerkeleyHumanoidMujocoEnvV2(gym.Env):
    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_actions = 12 
        self.num_obs = 37 # THE TRUTH: 19 pos + 18 vel

        self.kp = 30.0
        self.kd = 2.0
        self.effort_limits = np.array([20.0, 20.0, 30.0, 30.0, 20.0, 5.0, 20.0, 20.0, 30.0, 30.0, 20.0, 5.0])
        self.nominal_qpos = np.array([0.0, 0.0, -0.4, 0.8, -0.4, 0.0, 0.0, 0.0, -0.4, 0.8, -0.4, 0.0])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.step_count = 0

    def step(self, action):
        # This clip action was added because it exists in the SB3 implementation wherin we defined self.action_space with the condition that the lower and upper limit on actions was -1.0 to 1.0
        clipped_action = np.clip(action,-1.0,1.0)
        target_q = clipped_action * 0.3 + self.nominal_qpos
        for _ in range(10):
            self.data.qfrc_applied[:] = 0.0
            tau = self.kp * (target_q - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
            self.data.qfrc_applied[6:] = np.clip(tau, -self.effort_limits, self.effort_limits)
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        if self.render_mode == "human" and self.viewer: self.viewer.sync()

        obs = self._get_obs()
        reward = 1.0
        terminated = bool(self.data.qpos[2] < 0.3)
        truncated = self.step_count >= 1000 
        
        return obs, reward, terminated, truncated, {}

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     mujoco.mj_resetData(self.model, self.data)
    #     self.data.qpos[7:] = self.nominal_qpos
    #     self.step_count = 0
    #     mujoco.mj_forward(self.model, self.data)
    #     return self._get_obs(), {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Set the root height (Z-axis) so it doesn't start underground
        # qpos[2] is the height. Let's start it at 0.9 meters.
        self.data.qpos[2] = 0.9 
        
        # 2. Set the joints to the nominal standing pose
        self.data.qpos[7:] = self.nominal_qpos
        
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Get copies of the raw arrays so we don't accidentally mutate the physics engine
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # 2. Apply the scaling to the specific segments without dropping anything
        # Linear velocity (indices 0, 1, 2 in qvel)
        qvel[:3] = qvel[:3] * 2.0 
        
        # Angular velocity (indices 3, 4, 5 in qvel)
        qvel[3:6] = qvel[3:6] * 0.25 
        
        # Joint velocities (indices 6 to 18 in qvel)
        qvel[6:] = qvel[6:] * 0.05 
        
        # Note: Joint positions (qpos[7:]) have a scale of 1.0, so no math is needed.
        # The base orientation quaternion (qpos[3:7]) is already normalized between -1 and 1.
        
        # 3. Concatenate and return the full 37-element array
        obs = np.concatenate([qpos, qvel])
        return obs