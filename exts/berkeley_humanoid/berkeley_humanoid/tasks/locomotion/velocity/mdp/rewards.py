import numpy as np
import mujoco

def feet_slide_reward(model, data, foot_body_ids, action):
    """
    Penalize feet sliding. Translated from Isaac Lab tasks/locomotion/velocity/mdp/rewards.py
    """
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