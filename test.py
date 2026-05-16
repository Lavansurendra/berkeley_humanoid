import math
import mujoco
import mujoco.viewer
import time
import numpy as np

# 1. Load your model
model = mujoco.MjModel.from_xml_path('environment/g1_scene.xml')
data = mujoco.MjData(model)

# 2. Launch the passive visual viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    prev_pos = np.zeros(29)

    # Close the viewer cleanly if the script loop ends
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[:] = prev_pos

        # ------------ Cyclical Thigh Pushing Logic -----------------

        
        # clear the force applied on each thigh and the pelvis from the previous step
        data.qfrc_applied[6] = 0.0 # left hip pitch joint
        data.qfrc_applied[12] = 0.0 # right hip pitch joint

        # calculate the angular position (pitch) of where the thighs should be at the current timestep for a walking gait
        left_thigh_angle = 30*np.cos(2*np.pi*(data.time/4))
        right_thigh_angle = 30*np.cos(2*np.pi*((data.time -2)/4))

        # calculate the force that should be applied at the current time on each thigh
            # NOTE: positive forces make the legs go backwards
            # NOTE: there is a slight time delay between the force being applied to the thighs that would cause them to swing and the force on the pelvis in the y direction and that allows there to be some ground clearnace before the thigh swing begins
        left_thigh_qfrc = 40 * (-np.sin(np.pi * (left_thigh_angle/60)))
        right_thigh_qfrc = 40 * (-np.sin(np.pi * (right_thigh_angle/60)))

        # set the force being applied for the current time on each thigh
        data.qfrc_applied[6] = left_thigh_qfrc # left hip pitch joint
        data.qfrc_applied[12] = right_thigh_qfrc # right hip pitch joint

        #  ------------ Cyclical Knee Pushing Logic -----------------
        
        # clear the force applied on each thigh and the pelvis from the previous step
        data.qfrc_applied[9] = 0.0 # left knee pitch joint
        data.qfrc_applied[15] = 0.0 # right knee pitch joint

        # calculate the angular position (pitch) of where the thighs should be at the current timestep for a walking gait
        left_knee_angle = 50*max(0, np.sin(2*np.pi*(data.time/4)))
        right_knee_angle = 50*max(0, np.sin(2*np.pi*((data.time - 2)/4)))


        # calculate the force that should be applied at the current time on each knee
            # NOTE: positive forces make the shin go backwards
        left_knee_qfrc = 20 * (-np.sin(np.pi * ((left_knee_angle - 30)/50)))
        right_knee_qfrc = 20 * (-np.sin(np.pi * ((right_knee_angle - 30)/50)))

        # set the force being applied for the current time on each knee
        data.qfrc_applied[9] = left_knee_qfrc  # left knee pitch joint
        data.qfrc_applied[15] = right_knee_qfrc  # right knee pitch joint

        prev_pos = data.qpos[7:]



        # # Evaluate your sine function using data.time
        # # Example: 2 Hz frequency driving the first actuator (index 0)
        # data.ctrl[0] = math.sin(2.0 * math.pi * 2.0 * data.time)

        # Advance the physics simulation by one timestep
        mujoco.mj_step(model, data)

        # Pick up user inputs from the GUI and update the visual scene
        viewer.sync()

        # Rudimentary timing to keep the simulation running in near real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)