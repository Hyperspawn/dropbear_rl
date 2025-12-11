#!/usr/bin/env python3
"""Check remote worker environment status and provide cleanup instructions."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_VENV = PROJECT_ROOT / "env_remote"
REMOTE_MARKER = REMOTE_VENV / ".remote_bootstrap_ok"


def check_venv_exists():
    """Check if remote venv exists."""
    if REMOTE_VENV.exists():
        print(f"✓ Remote venv exists: {REMOTE_VENV}")
        return True
    else:
        print(f"✗ Remote venv NOT found: {REMOTE_VENV}")
        return False


def check_packages():
    """Check if required packages are installed in venv."""
    if not REMOTE_VENV.exists():
        return False

    python_exe = REMOTE_VENV / "bin" / "python"
    if not python_exe.exists():
        python_exe = REMOTE_VENV / "Scripts" / "python.exe"

    if not python_exe.exists():
        print("✗ Python executable not found in venv")
        return False

    packages = {
        "torch": "import torch; print(f'torch {torch.__version__}')",
        "gymnasium": "import gymnasium; print(f'gymnasium {gymnasium.__version__}')",
        "numpy": "import numpy; print(f'numpy {numpy.__version__}')",
        "rsl_rl": "import rsl_rl; print('rsl_rl OK')",
        "dropbear_rl_lab.remote": "import dropbear_rl_lab.remote; print('dropbear_rl_lab.remote OK')",
    }

    all_ok = True
    for pkg, test_code in packages.items():
        try:
            result = subprocess.run(
                [str(python_exe), "-c", test_code],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"✗ {pkg} NOT installed or import failed")
            all_ok = False

    # Check that IsaacLab is NOT installed
    try:
        subprocess.run(
            [str(python_exe), "-c", "import isaaclab"],
            capture_output=True,
            check=True
        )
        print("✗ WARNING: IsaacLab IS installed (should NOT be on remote worker)")
        all_ok = False
    except subprocess.CalledProcessError:
        print("✓ IsaacLab NOT installed (correct)")

    return all_ok


def check_torch_cuda():
    """Check if PyTorch CUDA is working."""
    if not REMOTE_VENV.exists():
        return False

    python_exe = REMOTE_VENV / "bin" / "python"
    if not python_exe.exists():
        python_exe = REMOTE_VENV / "Scripts" / "python.exe"

    if not python_exe.exists():
        return False

    test_code = """
import torch
if not torch.cuda.is_available():
    exit(1)
print(f"CUDA: {torch.version.cuda}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  {i}: {torch.cuda.get_device_name(i)}")
"""

    try:
        result = subprocess.run(
            [str(python_exe), "-c", test_code],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ PyTorch CUDA:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
        return True
    except subprocess.CalledProcessError:
        print("✗ PyTorch CUDA NOT working")
        return False


def main():
    print("=" * 70)
    print("Remote A100 Worker Environment Status")
    print("=" * 70)
    print()

    venv_exists = check_venv_exists()
    print()

    if venv_exists:
        print("Checking packages...")
        packages_ok = check_packages()
        print()

        print("Checking PyTorch CUDA...")
        cuda_ok = check_torch_cuda()
        print()

        print("=" * 70)
        if packages_ok and cuda_ok:
            print("✓ Environment is ready!")
            print("=" * 70)
            print()
            print("You can now run:")
            print("  python3 remote.py --app-address=<controller_nkn_address>")
            return 0
        else:
            print("✗ Environment has issues - needs rebuild")
            print("=" * 70)
    else:
        print("=" * 70)
        print("✗ Environment not found - needs bootstrap")
        print("=" * 70)

    print()
    print("To fix, run these commands on the A100 worker:")
    print()
    print("  # Clean old environment")
    print("  rm -rf env_remote/")
    print()
    print("  # Bootstrap fresh environment")
    print("  python3 remote.py --app-address=<controller_nkn_address>")
    print()
    print("This will:")
    print("  - Auto-detect CUDA")
    print("  - Install PyTorch with CUDA support")
    print("  - Install rsl-rl-lib")
    print("  - Install dropbear_rl_lab[remote] (gymnasium, numpy)")
    print("  - Verify NO IsaacLab")
    print("  - Test PyTorch GPU operations")
    print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
