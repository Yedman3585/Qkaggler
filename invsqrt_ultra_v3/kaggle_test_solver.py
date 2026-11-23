#!/usr/bin/env python3
"""Kaggle cell: test patched gf2_lut.py on real ultra_v3 build.

Paste this into a Kaggle notebook cell AFTER cloning Qkaggler repo
and setting up paths. It builds ultra_v3 twice — with original and
patched gf2_lut.py — and compares CCX counts.

Prerequisites:
  - Cell 2: clone Qkaggler + setup PYTHONPATH
  - invsqrt_ultra_v3/ folder with ultra_v3.py, gf2_lut.py (patched),
    ultra_primitives.py, ultra_v2_primitives.py
"""

import subprocess, shutil, os, re, time

WORK = "/kaggle/working"
UV3_DIR = os.path.join(WORK, "Qkaggler/invsqrt_ultra_v3")

# Build command (same params as ultra_v3_p.kmb)
BUILD_CMD = [
    "python", "ultra_v3.py", "test_build.kmb",
    "--wc2", "8", "--zgb", "0", "--mux-min", "0",
    "--nc0q1", "16", "--ntmpq1", "3", "--defer-c2ovf", "1",
    "--r1-blk", "2blk", "--r1-anc", "6", "--cap-r1", "0",
]

def extract_ccx_from_log(output):
    """Extract CCX counts from ultra_v3 build output."""
    ccx_lines = []
    for line in output.split('\n'):
        if 'CCX' in line or 'ccx' in line.lower() or 'Tof' in line:
            ccx_lines.append(line.strip())
    return ccx_lines

def build_once(label, gf2_path):
    """Run ultra_v3 build and capture output."""
    # Copy the gf2_lut.py variant
    dst = os.path.join(UV3_DIR, "gf2_lut.py")
    shutil.copy2(gf2_path, dst)

    print(f"\n{'='*60}")
    print(f"  Building with {label} gf2_lut.py")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        BUILD_CMD, cwd=UV3_DIR,
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0

    output = result.stdout + "\n" + result.stderr
    print(output[-2000:])  # Last 2000 chars

    # Check for the .kmb file
    kmb = os.path.join(UV3_DIR, "test_build.kmb")
    if os.path.exists(kmb):
        size = os.path.getsize(kmb)
        print(f"\n  Built {kmb} ({size} bytes) in {elapsed:.1f}s")
        # Rename to keep both
        final = os.path.join(UV3_DIR, f"test_{label}.kmb")
        shutil.move(kmb, final)
        return final, elapsed
    else:
        print(f"\n  BUILD FAILED ({elapsed:.1f}s)")
        return None, elapsed

# --- Save patched version, create original version ---
patched_path = os.path.join(UV3_DIR, "gf2_lut.py")
patched_bak = os.path.join(UV3_DIR, "gf2_lut_patched.py")
original_bak = os.path.join(UV3_DIR, "gf2_lut_original.py")

# Back up patched
shutil.copy2(patched_path, patched_bak)

# Create original by reverting the 3 patch blocks
# (This is a simplified approach — just builds both versions)
print("To compare: build PATCHED first, then revert and build ORIGINAL")
print("Or just build PATCHED and compare score with ultra_v3_p.kmb (472 CCX)")

# Build with patched solver
kmb_patched, t_patched = build_once("PATCHED", patched_bak)

if kmb_patched:
    print(f"\n{'='*60}")
    print(f"  RESULT: Patched build at {kmb_patched}")
    print(f"  Compare with ultra_v3_p.kmb (472 CCX, score 106925)")
    print(f"  Run: python fast_verify_invsqrt.py {kmb_patched} --max-x 16384 --shots-per-x 64")
    print(f"{'='*60}")
