#!/usr/bin/env python3
# Copyright (c) 2025, Hyperspawn Robotics.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quick test to verify Dropbear environments can be created without errors."""

def test_environment_creation():
    """Test that environments can be created without configuration errors."""
    try:
        import gymnasium as gym
        import dropbear_rl_lab.tasks  # Register environments
        
        print("✅ Successfully imported dropbear_rl_lab.tasks")
        
        # Test that environments are registered
        all_envs = list(gym.envs.registry.keys())
        dropbear_envs = [env_id for env_id in all_envs if "Dropbear" in env_id]
        
        print(f"✅ Found {len(dropbear_envs)} Dropbear environments:")
        for env_id in sorted(dropbear_envs):
            print(f"   - {env_id}")
        
        # Test configuration loading (without creating actual environment)
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        
        # Test training environment config
        try:
            train_cfg = load_cfg_from_registry("Isaac-Velocity-Dropbear-v0", "env_cfg_entry_point")
            print("✅ Training environment configuration loads successfully")
        except Exception as e:
            print(f"❌ Training environment config error: {e}")
            return False
        
        # Test play environment config
        try:
            play_cfg = load_cfg_from_registry("Isaac-Velocity-Dropbear-Play-v0", "play_env_cfg_entry_point")
            print("✅ Play environment configuration loads successfully")
        except Exception as e:
            print(f"❌ Play environment config error: {e}")
            return False
            
        print("\n🎉 All configuration tests passed!")
        print("\nReady to use commands:")
        print("  C:\\isaac-sim\\python.bat scripts/rsl_rl/train.py --task Isaac-Velocity-Dropbear-v0 --max_iterations 1")
        print("  C:\\isaac-sim\\python.bat scripts/rsl_rl/play.py --task Isaac-Velocity-Dropbear-Play-v0 --video")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_environment_creation()
