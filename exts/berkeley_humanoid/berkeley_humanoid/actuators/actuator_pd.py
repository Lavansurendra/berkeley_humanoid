# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
# MuJoCo Conversion Version

import numpy as np

class MujocoPDActuator:
    """
    MuJoCo implementation of the IdentifiedActuator for Berkeley Humanoid.
    Translates the original PyTorch-based friction model into Numpy for MuJoCo's control loop.
    """
    def __init__(self, stiffness, damping, friction_static=0.0, friction_dynamic=0.0, activation_vel=0.1, effort_limit=100.0):
        # PD Gains
        self.stiffness = np.array(stiffness, dtype=np.float32)
        self.damping = np.array(damping, dtype=np.float32)
        
        # Friction parameters (from BERKELEY_HUMANOID_HXX_ACTUATOR_CFG etc.)
        self.friction_static = np.array(friction_static, dtype=np.float32)
        self.friction_dynamic = np.array(friction_dynamic, dtype=np.float32)
        self.activation_vel = activation_vel
        self.effort_limit = effort_limit

    def compute_torques(self, target_positions, current_positions, current_velocities):
        """
        Computes the joint torques to apply to MuJoCo based on PD control and the custom friction model.
        """
        # 1. Standard PD Control calculation
        position_error = target_positions - current_positions
        velocity_error = 0.0 - current_velocities # Assuming target velocity is always 0 for position control
        
        pd_torques = (self.stiffness * position_error) + (self.damping * velocity_error)
        
        # 2. Apply the custom friction model (Translated from original PyTorch code)
        # Using np.tanh exactly as the original used torch.tanh
        friction_torques = (self.friction_static * np.tanh(current_velocities / self.activation_vel)) + \
                           (self.friction_dynamic * current_velocities)
                           
        # 3. Final torques (PD minus friction)
        final_torques = pd_torques - friction_torques
        
        # 4. Clip to effort limits
        final_torques = np.clip(final_torques, -self.effort_limit, self.effort_limit)
        
        return final_torques