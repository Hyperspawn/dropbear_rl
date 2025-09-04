# Changelog

All notable changes to the Dropbear RL Lab project will be documented in this file.

## [0.1.0] - 2025-01-05

### Added
- Initial release of Dropbear RL Lab
- Complete Isaac Lab integration for Dropbear humanoid robot
- Velocity tracking environment with terrain support
- PPO training configuration optimized for humanoid locomotion
- Comprehensive documentation and examples
- Professional README with installation and usage instructions
- Apache 2.0 license for open source compliance

### Features
- **Robot Configuration**: Complete Dropbear robot model with 22 DoF
- **Training Environment**: `Isaac-Velocity-Dropbear-v0` for policy training
- **Play Environment**: `Isaac-Velocity-Dropbear-Play-v0` for policy evaluation
- **Modular MDP**: Configurable rewards, observations, and commands
- **GPU Acceleration**: Support for thousands of parallel environments
- **Video Recording**: Built-in video capture for training and evaluation

### Technical Details
- Based on Isaac Lab 2.2.0 and Isaac Sim 5.0.0
- Uses RSL-RL PPO implementation with humanoid-optimized parameters
- Supports both flat terrain and complex terrain generation
- Includes contact sensing and IMU feedback
- Configurable joint actuators with proper limits and dynamics

### Fixed
- Environment registration issues with missing `play_env_cfg_entry_point`
- AttributeError in PLAY configuration when terrain_generator is None
- Import path issues and package installation problems
- Documentation clarity and command examples

### Confirmed Working Commands
```bash
# Training (1 iteration test)
C:\isaac-sim\python.bat scripts\rsl_rl\train.py --task Isaac-Velocity-Dropbear-v0 --max_iterations 1

# Playing with video recording  
C:\isaac-sim\python.bat scripts\rsl_rl\play.py --task Isaac-Velocity-Dropbear-Play-v0 --video
```

### Known Limitations
- Some utility scripts may hang during Isaac Sim initialization
- Requires NVIDIA GPU with CUDA support
- Large memory requirements for high environment counts

---

**Developed by Hyperspawn Robotics 🤗**
