import math
import mujoco
import mujoco.viewer
import time
import numpy as np

# 1. Load your model
model = mujoco.MjModel.from_xml_path('environment/g1_scene.xml')
data = mujoco.MjData(model)

# Map qfrc_applied indices to ctrl indices
# (Assuming 6->0, 9->3, 12->6, 15->9 based on standard free-joint offset)
left_hip_pitch = 0
left_knee_pitch = 3
right_hip_pitch = 6
right_knee_pitch = 9

# 2. Launch the passive visual viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    nominal_ctrl = np.zeros(29)

    while viewer.is_running():
        step_start = time.time()

        data.ctrl[:] = nominal_ctrl

        data.xfrc_applied[0, 1] = 0.0
        data.xfrc_applied[0, 2] = 0.0
        data.xfrc_applied[0, 4] = 0.0
        data.xfrc_applied[0, 1] = -50 * data.qpos[1] * (1 - (data.time / 15))
        data.xfrc_applied[0, 2] = 350 * (1 - (data.time / 15))
        data.xfrc_applied[0, 4] = -50 * np.arccos(np.dot(np.array([1,0,0,0]), data.qpos[3:7])) * (1 - (data.time / 15))


        # ------------ Cyclical Thigh Position Logic -----------------
        left_thigh_angle_deg = 30 * np.cos(2 * np.pi * (data.time / 4))
        right_thigh_angle_deg = 30 * np.cos(2 * np.pi * ((data.time - 2) / 4))

        # deg2rad should convert the angle to radians which is what ctrl takes as input
        data.ctrl[left_hip_pitch] = np.deg2rad(left_thigh_angle_deg)
        data.ctrl[right_hip_pitch] = np.deg2rad(right_thigh_angle_deg)

        # ------------ Cyclical Knee Position Logic -----------------
        # Calculate the angular position (pitch) in degrees, then convert to radians
        left_knee_angle_deg = 50 * max(0, np.sin(2 * np.pi * (data.time / 4)))
        right_knee_angle_deg = 50 * max(0, np.sin(2 * np.pi * ((data.time - 2) / 4)))

        # deg2rad should convert the angle to radians which is what ctrl takes as input
        data.ctrl[left_knee_pitch] = np.deg2rad(left_knee_angle_deg)
        data.ctrl[right_knee_pitch] = np.deg2rad(right_knee_angle_deg)

        # Advance the physics simulation by one timestep
        mujoco.mj_step(model, data)

        # Pick up user inputs from the GUI and update the visual scene
        viewer.sync()

        # Rudimentary timing to keep the simulation running in near real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)