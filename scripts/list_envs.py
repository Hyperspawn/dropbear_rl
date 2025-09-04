#!/usr/bin/env python3
# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to list all registered Dropbear RL Lab environments."""

def main():
    """List all registered Dropbear environments."""
    print("\n" + "="*50)
    print("DROPBEAR RL LAB - REGISTERED ENVIRONMENTS")  
    print("="*50)
    
    try:
        import gymnasium as gym
        # Import to register environments
        import dropbear_rl_lab.tasks  # noqa: F401
        
        # Get all registered environments
        all_envs = list(gym.envs.registry.keys())
        
        # Filter for Dropbear environments
        dropbear_envs = [env_id for env_id in all_envs if "Dropbear" in env_id]
        
        if dropbear_envs:
            print(f"\nFound {len(dropbear_envs)} Dropbear environment(s):")
            for i, env_id in enumerate(sorted(dropbear_envs), 1):
                print(f"  {i:2d}. {env_id}")
            
            print("\nWorking commands:")
            print("  # Training:")
            print("  C:\\isaac-sim\\python.bat scripts/rsl_rl/train.py --task Isaac-Velocity-Dropbear-v0 --max_iterations 1")
            print("  # Playing (requires trained model):")
            print("  C:\\isaac-sim\\python.bat scripts/rsl_rl/play.py --task Isaac-Velocity-Dropbear-Play-v0")
        else:
            print("\nNo Dropbear environments found!")
            print("Make sure dropbear_rl_lab package is properly installed.")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Isaac Lab and dropbear_rl_lab are properly installed.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()