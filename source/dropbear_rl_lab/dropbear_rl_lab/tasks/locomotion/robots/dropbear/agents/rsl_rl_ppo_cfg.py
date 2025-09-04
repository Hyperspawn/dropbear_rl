# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PPO configuration for Dropbear humanoid robot velocity tracking.

This module contains the PPO (Proximal Policy Optimization) configuration
specifically tuned for the Dropbear humanoid robot's velocity tracking task.
The parameters are optimized for stable learning of bipedal locomotion.
"""

from isaaclab.utils import configclass

from dropbear_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class DropbearVelocityRoughPPORunnerCfg(BasePPORunnerCfg):
    """PPO configuration for Dropbear velocity tracking on rough terrain.
    
    This configuration is specifically tuned for the Dropbear humanoid robot
    to learn stable bipedal locomotion with velocity tracking capabilities.
    The parameters are optimized for the complexity of humanoid control.
    
    Key Features:
        - Larger neural networks to handle humanoid complexity
        - Conservative learning rates for stability
        - Longer rollouts for better temporal understanding
        - Adaptive learning schedule for robust training
    """
    
    # Experiment identification
    experiment_name: str = "dropbear_velocity"
    """Name of the experiment for logging and identification."""
    
    # Training parameters optimized for humanoid complexity
    num_steps_per_env: int = 32
    """Number of steps per environment per rollout (longer for humanoid)."""
    
    max_iterations: int = 30000
    """Maximum number of training iterations."""
    
    def __post_init__(self):
        """Post-initialization to set up network and algorithm parameters."""
        super().__post_init__()
        
        # Neural network architecture - larger for complex humanoid control
        self.policy.init_noise_std = 0.5  # Conservative initial exploration
        self.policy.actor_hidden_dims = [512, 512, 256]  # Large actor network
        self.policy.critic_hidden_dims = [512, 512, 256]  # Large critic network
        self.policy.activation = "elu"  # ELU activation for smooth gradients
        
        # PPO algorithm parameters tuned for humanoid stability
        self.algorithm.value_loss_coef = 1.0  # Standard value loss coefficient
        self.algorithm.use_clipped_value_loss = True  # Clip value loss for stability
        self.algorithm.clip_param = 0.2  # Conservative clipping for stable updates
        self.algorithm.entropy_coef = 0.01  # Moderate entropy for exploration
        self.algorithm.num_learning_epochs = 3  # Fewer epochs to prevent overfitting
        self.algorithm.num_mini_batches = 4  # Mini-batch size for gradient updates
        self.algorithm.learning_rate = 1.0e-4  # Conservative learning rate
        self.algorithm.schedule = "adaptive"  # Adaptive learning rate schedule
        self.algorithm.gamma = 0.99  # Standard discount factor
        self.algorithm.lam = 0.95  # GAE lambda for advantage estimation
        self.algorithm.desired_kl = 0.01  # Conservative KL divergence target
        self.algorithm.max_grad_norm = 0.5  # Gradient clipping for stability
