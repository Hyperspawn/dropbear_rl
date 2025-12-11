#!/bin/bash
# Clean and reset remote environment for A100 workers

set -e

echo "=============================================="
echo "Cleaning Remote A100 Worker Environment"
echo "=============================================="
echo ""

# Remove old environment
if [ -d "env_remote" ]; then
    echo "[1/3] Removing old env_remote/ directory..."
    rm -rf env_remote/
    echo "      ✓ Removed env_remote/"
else
    echo "[1/3] No env_remote/ directory found (clean)"
fi

# Remove marker file
if [ -f "env_remote/.remote_bootstrap_ok" ]; then
    echo "[2/3] Removing bootstrap marker..."
    rm -f env_remote/.remote_bootstrap_ok
    echo "      ✓ Removed marker"
else
    echo "[2/3] No bootstrap marker found (clean)"
fi

# Keep NKN seed for persistence
if [ -f ".remote_nkn_seed" ]; then
    echo "[3/3] Preserving NKN seed (.remote_nkn_seed)"
    echo "      ✓ NKN address will remain the same"
else
    echo "[3/3] No NKN seed found (new one will be generated)"
fi

echo ""
echo "=============================================="
echo "Environment cleaned successfully!"
echo "=============================================="
echo ""
echo "Next step: Run remote.py to bootstrap clean environment"
echo ""
echo "  python3 remote.py --app-address=<controller_nkn_address>"
echo ""
echo "This will install:"
echo "  - PyTorch 2.7.0 (CUDA 12.8)"
echo "  - rsl-rl-lib >= 2.3.1"
echo "  - dropbear_rl_lab[remote] (includes gymnasium, numpy)"
echo "  - NO Isaac Sim"
echo "  - NO IsaacLab"
echo ""
