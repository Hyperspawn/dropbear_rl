# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Configuration for Dropbear humanoid robot.

This module defines the complete configuration for the Dropbear humanoid robot,
including physical properties, actuators, initial states, and joint mappings.
The configuration is optimized for reinforcement learning tasks in Isaac Lab.
"""

import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class DropbearArticulationCfg(ArticulationCfg):
    """Configuration class for Dropbear humanoid robot articulations.
    
    This configuration extends the base ArticulationCfg to include Dropbear-specific
    parameters and joint mappings for the humanoid robot.
    
    Attributes:
        joint_sdk_names: List of joint names for SDK integration.
        soft_joint_pos_limit_factor: Safety factor for joint position limits.
    """

    joint_sdk_names: Optional[list[str]] = None
    """List of joint names used for SDK integration and control mapping."""

    soft_joint_pos_limit_factor: float = 0.9
    """Factor to apply to joint limits for safety (0.9 = 90% of max range)."""


# Get the model directory relative to the current file location
DROPBEAR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "dropbear_model")
DROPBEAR_MODEL_DIR = os.path.abspath(DROPBEAR_MODEL_DIR)

# Dropbear humanoid robot configuration
DROPBEAR_CFG = DropbearArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DROPBEAR_MODEL_DIR}/Dropbear/usd/dropbear.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # Disable gravity to prevent physics explosions
            retain_accelerations=False,
            linear_damping=0.1,  # Add some damping for stability
            angular_damping=0.1,  # Add some damping for stability
            max_linear_velocity=100.0,  # Reduce max velocities
            max_angular_velocity=100.0,  # Reduce max velocities
            max_depenetration_velocity=1.0,  # Reduce depenetration velocity
            enable_gyroscopic_forces=False,  # Disable for simplicity
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collisions for stability
            solver_position_iteration_count=4,  # Reduce iterations for stability
            solver_velocity_iteration_count=0,  # Reduce for stability
            # Enable debugging joint to fix robot to world
            fix_root_link=True,  # Disable debugging_joint to prevent coordinate explosions
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # Left arm joints - neutral position for balance
            "LH_yaw": 0.0,
            "LH_pitch": 0.0,
            "LH_roll": 0.0,
            "LH_Revolute41": -0.5,
            "LH_wrist_roll": 0.0,
            # Right arm joints - neutral position for balance
            "RH_yaw": 0.0,
            "RH_pitch": 0.0,
            "RH_roll": 0.0,
            "RH_Revolute41": -0.5,
            "RH_wrist_roll": 0.0,
            # Pelvic girdle - neutral stance
            "PG_left_leg_pitch": 0.0,
            "PG_left_leg_roll": 0.0,
            "PG_right_leg_pitch": 0.0,
            "PG_right_leg_roll": 0.0,
            # Left leg - standing position
            "LL_hip_joint": -0.2,
            "LL_knee_actuator_joint": 0.4,
            "LL_Revolute28": -0.2,
            "LL_Revolute29": 0.0,
            # Right leg - standing position
            "RL_hip_joint": -0.2,
            "RL_knee_actuator_joint": 0.4,
            "RL_Revolute28": -0.2,
            "RL_Revolute29": 0.0,
            # Head/neck - neutral position
            "head_LeadScrew1": 0.0,
            "head_LeadScrew2": 0.0,
            "head_LeadScrew3": 0.0,
            "head_LeadScrew4": 0.0,
            "head_LeadScrew5": 0.0,
            "head_LeadScrew6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[
                # Left arm actuators
                "LH_yaw", "LH_pitch", "LH_roll", "LH_Revolute41", "LH_wrist_roll",
                # Right arm actuators
                "RH_yaw", "RH_pitch", "RH_roll", "RH_Revolute41", "RH_wrist_roll",
                # Pelvic girdle actuators
                "PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll",
                # Leg actuators
                "LL_hip_joint", "LL_knee_actuator_joint", "RL_hip_joint", "RL_knee_actuator_joint",
                "LL_Revolute28", "LL_Revolute29", "RL_Revolute28", "RL_Revolute29",
            ],
            effort_limit_sim=80.0,  # Maximum torque (Nm) - reasonable for humanoid
            velocity_limit_sim=50.0,  # Maximum angular velocity (rad/s)
            stiffness=40.0,  # PD controller proportional gain - reduced for stability
            damping=2.0,  # PD controller derivative gain - reduced for stability
            friction=0.01,  # Joint friction coefficient - reduced
            armature=0.01,  # Joint armature (inertia)
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
                "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6",
            ],
            effort_limit_sim=50.0,  # Lower torque limit for neck
            velocity_limit_sim=50.0,  # Lower velocity limit for neck
            stiffness=100.0,  # Higher stiffness for precise head control
            damping=5.0,  # Higher damping for stability
            friction=0.01,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        # Left arm joints
        "LH_yaw", "LH_pitch", "LH_roll", "LH_Revolute41", "LH_wrist_roll",
        # Right arm joints
        "RH_yaw", "RH_pitch", "RH_roll", "RH_Revolute41", "RH_wrist_roll",
        # Pelvic girdle joints
        "PG_left_leg_pitch", "PG_left_leg_roll", "PG_right_leg_pitch", "PG_right_leg_roll",
        # Leg joints
        "LL_hip_joint", "LL_knee_actuator_joint", "RL_hip_joint", "RL_knee_actuator_joint",
        "LL_Revolute28", "LL_Revolute29", "RL_Revolute28", "RL_Revolute29",
        # Head/neck joints
        "head_LeadScrew1", "head_LeadScrew2", "head_LeadScrew3",
        "head_LeadScrew4", "head_LeadScrew5", "head_LeadScrew6",
    ],
)

# Export the main configuration
__all__ = ["DROPBEAR_CFG", "DropbearArticulationCfg"]