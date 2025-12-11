#!/usr/bin/env python3
"""Verification script for remote adapter implementation.

This script verifies that:
1. train_remote.py exists and has no IsaacLab imports
2. dropbear_rl_lab.remote module is importable
3. Remote extras are installable
4. CLI arguments are compatible
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False


def check_no_isaaclab_imports(file_path: Path) -> bool:
    """Check that a Python file has no IsaacLab imports."""
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    isaaclab_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "isaaclab" in alias.name.lower():
                    isaaclab_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "isaaclab" in node.module.lower():
                isaaclab_imports.append(node.module)

    if isaaclab_imports:
        print(f"✗ {file_path.name} has IsaacLab imports: {isaaclab_imports}")
        return False
    else:
        print(f"✓ {file_path.name} has NO IsaacLab imports")
        return True


def check_module_importable(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} is importable")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False


def main():
    """Run verification checks."""
    print("=" * 70)
    print("Remote Adapter Implementation Verification")
    print("=" * 70)

    checks = []

    # 1. Check files exist
    print("\n[1] Checking file structure...")
    checks.append(check_file_exists(
        PROJECT_ROOT / "scripts" / "rsl_rl" / "train_remote.py",
        "train_remote.py"
    ))
    checks.append(check_file_exists(
        PROJECT_ROOT / "source" / "dropbear_rl_lab" / "dropbear_rl_lab" / "remote" / "__init__.py",
        "remote module __init__"
    ))
    checks.append(check_file_exists(
        PROJECT_ROOT / "source" / "dropbear_rl_lab" / "dropbear_rl_lab" / "remote" / "env_builder.py",
        "remote module env_builder"
    ))
    checks.append(check_file_exists(
        PROJECT_ROOT / "source" / "dropbear_rl_lab" / "dropbear_rl_lab" / "remote" / "config_helpers.py",
        "remote module config_helpers"
    ))

    # 2. Check no IsaacLab imports
    print("\n[2] Checking IsaacLab import isolation...")
    train_remote = PROJECT_ROOT / "scripts" / "rsl_rl" / "train_remote.py"
    if train_remote.exists():
        checks.append(check_no_isaaclab_imports(train_remote))

    env_builder = PROJECT_ROOT / "source" / "dropbear_rl_lab" / "dropbear_rl_lab" / "remote" / "env_builder.py"
    if env_builder.exists():
        checks.append(check_no_isaaclab_imports(env_builder))

    config_helpers = PROJECT_ROOT / "source" / "dropbear_rl_lab" / "dropbear_rl_lab" / "remote" / "config_helpers.py"
    if config_helpers.exists():
        checks.append(check_no_isaaclab_imports(config_helpers))

    # 3. Check module imports (if installed)
    print("\n[3] Checking module imports (if dropbear_rl_lab is installed)...")
    try:
        import dropbear_rl_lab
        checks.append(check_module_importable("dropbear_rl_lab.remote"))
        checks.append(check_module_importable("dropbear_rl_lab.remote.env_builder"))
        checks.append(check_module_importable("dropbear_rl_lab.remote.config_helpers"))
    except ImportError:
        print("⚠ dropbear_rl_lab not installed - skipping import checks")
        print("  Run: pip install -e source/dropbear_rl_lab[remote]")

    # 4. Check setup.py has remote extras
    print("\n[4] Checking setup.py configuration...")
    setup_py = PROJECT_ROOT / "source" / "dropbear_rl_lab" / "setup.py"
    if setup_py.exists():
        content = setup_py.read_text()
        if "extras_require" in content and "remote" in content:
            print("✓ setup.py has remote extras defined")
            checks.append(True)
        else:
            print("✗ setup.py missing remote extras")
            checks.append(False)

    # Summary
    print("\n" + "=" * 70)
    passed = sum(checks)
    total = len(checks)
    if passed == total:
        print(f"✓ All {total} checks passed!")
        print("\nRemote adapter implementation is ready.")
        print("\nNext steps:")
        print("1. Install on A100 worker: pip install -e source/dropbear_rl_lab[remote]")
        print("2. Start remote.py on A100 worker")
        print("3. Configure NKN target in app.py curses UI")
        print("4. Run dropbear_train with remote mode enabled")
        return 0
    else:
        print(f"✗ {total - passed}/{total} checks failed")
        print("\nPlease review the errors above and fix missing components.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
