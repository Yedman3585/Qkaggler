#!/usr/bin/env python3
"""Quick test: compare original vs patched _solve_bcast_quad_multi.

Run from the invsqrt_ultra_v3 directory:
    python test_solver_improvement.py

Requires: numpy, kickmix (or just tests the pure-Python path without Numba).
"""

import random
import time
import sys
import os

# Temporarily disable Numba to test pure-Python path
# Comment out the next line to test with Numba
# os.environ['NUMBA_DISABLE_JIT'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll import and monkey-patch to compare original vs improved
import numpy as np

# Generate realistic test data: M wires with n_in=256 bit truth tables
def make_test_case(M, k_targets, n_in=256, seed=42):
    """Create a test case similar to what ultra_v3 LUT stages produce."""
    rng = random.Random(seed)
    # Random wire truth tables
    W_vals = [rng.getrandbits(n_in) for _ in range(M)]
    # Some broadcast products (simulating already-computed values)
    prod_vals = []
    for i in range(min(M, 5)):
        for j in range(i+1, min(M, 5)):
            prod_vals.append(W_vals[i] & W_vals[j])

    # Random target truth tables (what we need to synthesize)
    targets = [rng.getrandbits(n_in) for _ in range(k_targets)]
    return W_vals, prod_vals, targets


def test_solver(label, M, k_targets, n_in=256, seed=42):
    """Test the solver and report CCX counts."""
    from gf2_lut import _solve_bcast_quad_multi, _verify_quad_eval, _iter_bits

    W_vals, prod_vals, targets = make_test_case(M, k_targets, n_in, seed)

    total_ccx = 0
    rng = random.Random(seed + 1000)
    t0 = time.time()

    remaining = list(targets)
    cur_val = 0

    for bit_idx in range(k_targets):
        if not remaining:
            break
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(
            W_vals, prod_vals, remaining, n_in, rng, trials=18, cur_val=cur_val
        )
        # Verify correctness
        # Build the target value from ct mask
        goal = cur_val
        for ti in range(len(remaining)):
            if ct & (1 << ti):
                goal ^= remaining[ti]

        # Verify the quadratic form matches
        recon = _verify_quad_eval(W_vals, p_uv, lm, cb, n_in)
        # Apply broadcast products
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= prod_vals[pi]
        recon ^= bcast_val

        FULL = (1 << n_in) - 1
        assert (recon ^ goal) == 0 or (recon ^ goal ^ FULL) == 0, \
            f"Verification FAILED at bit {bit_idx}!"

        n_ccx = len(p_uv)
        total_ccx += n_ccx

        # Remove synthesized target and add new wire
        new_remaining = []
        for ti in range(len(remaining)):
            if not (ct & (1 << ti)):
                new_remaining.append(remaining[ti])
            else:
                # This target is now a wire
                syn_val = _verify_quad_eval(W_vals, p_uv, lm, cb, n_in) ^ bcast_val
                W_vals = list(W_vals) + [syn_val]
                prod_vals = list(prod_vals)  # Could add new products
        remaining = new_remaining
        cur_val = 0

    elapsed = time.time() - t0
    print(f"  {label}: M={M}, k={k_targets} → {total_ccx} CCX total ({elapsed:.2f}s)")
    return total_ccx


def main():
    print("=" * 60)
    print("GF(2) LUT Solver Improvement Test")
    print("=" * 60)

    # Test cases matching ultra_v3 LUT stages
    test_configs = [
        ("EarlyW0 (c0_hi)", 6, 13, 64),    # n_in=64 for this stage
        ("QROM1 (c1+c0_lo)", 8, 34, 256),
        ("c2fwd", 34, 8, 256),
        ("c0r", 30, 4, 256),
        ("c2_rev", 23, 8, 256),
        ("addr_rev", 26, 8, 256),
    ]

    total = 0
    for label, M, k, n_in in test_configs:
        ccx = test_solver(label, M, k, n_in)
        total += ccx

    print(f"\n  TOTAL: {total} CCX")
    print(f"  (Original ultra_v3_p: 472 CCX)")
    print(f"  (v34 target: ~340 CCX)")
    print(f"  Savings: {472 - total} CCX")


if __name__ == '__main__':
    main()
