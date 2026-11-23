#!/bin/bash
# Run df_init.py with correct environment
# Usage: bash run_df_init.sh [R B C] [--fit N]
# Default: R=5 B=80 C=30 --fit 3000

set -e

source ~/venv-jax/bin/activate

export DFTHC_DATA="/mnt/c/Users/user/Downloads/google-verifiable-quantum-circuit-compilation"
export DFTHC_STARTER="/mnt/c/Users/user/Desktop/quantum_kaggle/starter-kit/starter_kit"

cd /mnt/c/Users/user/Desktop/quantum_kaggle/Qkaggler/optimizer

# Check verifiers exist
if [ ! -f "$DFTHC_DATA/verifiers/dfthcsos/FeMoCo76o113e.fcidump" ]; then
    echo "ERROR: FCIDUMP not found at $DFTHC_DATA/verifiers/dfthcsos/"
    echo "Looking for it..."
    find /mnt/c/Users/user -name "FeMoCo76o113e.fcidump" -type f 2>/dev/null | head -5
    exit 1
fi

R=${1:-5}
B=${2:-80}
C=${3:-30}
FIT=${4:---fit}
FITN=${5:-3000}

echo "=== df_init.py for R=$R, B=$B, C=$C ==="
echo "DFTHC_DATA=$DFTHC_DATA"
echo ""

python experiments/df_init.py "seed_${R}x${B}x${C}.npz" $R $B $C $FIT $FITN

echo ""
echo "=== Done! Output: seed_${R}x${B}x${C}.npz ==="
