"""GPU job: shrink ahoirg factorization to smaller shape, then finish pipeline.

Usage on Kaggle:
    python gpu_shrink_job.py inputs/ahoirg_9x143x42.npz --target-B 100 --target-C 30

This runs:
1. shrink.py: 143x42 → target shape with many L-BFGS iters (GPU accelerated)
2. run.py --start shrunk.npz recipes/finish_shrunk.toml: quantize → refine → refit → build
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run_cmd(cmd, label=""):
    print(f"\n{'='*60}", flush=True)
    print(f"== {label}: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, '-u', *cmd])
    dt = time.time() - t0
    print(f"== {label} {'OK' if r.returncode == 0 else 'FAILED'} in {dt:.0f}s", flush=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path, help='pre-fitted NPZ (e.g. ahoirg_9x143x42.npz)')
    ap.add_argument('--target-B', type=int, default=100)
    ap.add_argument('--target-C', type=int, default=30)
    ap.add_argument('--kappa', type=float, default=5e-5)
    ap.add_argument('--fade', type=int, default=500)
    ap.add_argument('--fit', type=int, default=5000, help='L-BFGS iters per intermediate step')
    ap.add_argument('--final', type=int, default=10000, help='L-BFGS iters for final target')
    ap.add_argument('--out', type=Path, default=Path('/kaggle/working/results'))
    ap.add_argument('--skip-pipeline', action='store_true', help='only shrink, skip quantize/refine/build')
    a = ap.parse_args()
    
    import numpy as np
    data = np.load(a.input)
    R, B_start, C_start = data['W'].shape
    print(f"Input shape: R={R}, B={B_start}, C={C_start}")
    
    a.out.mkdir(parents=True, exist_ok=True)
    
    # Build shrink schedule: gradual steps of ~5-10 in B and ~2-3 in C
    schedule = []
    B, C = B_start, C_start
    while B > a.target_B or C > a.target_C:
        B_step = max(a.target_B, B - 10)
        C_step = max(a.target_C, C - 3)
        # Don't shrink B below target while C is still above
        if C > a.target_C and B_step <= a.target_B:
            B_step = max(a.target_B, B)
        B, C = B_step, C_step
        schedule.append(f"{B}x{C}")
    
    if not schedule:
        print("Already at target shape!")
        return
    
    schedule_str = ','.join(schedule)
    print(f"Shrink schedule: {schedule_str}")
    
    # Run shrink.py
    shrink_out = a.out / 'shrink_out'
    shrink_out.mkdir(exist_ok=True)
    
    shrink_cmd = [
        str(HERE / 'experiments' / 'shrink.py'),
        str(a.input),
        '--schedule', schedule_str,
        '--save', schedule[-1],  # save the final target
        '--kappa', str(a.kappa),
        '--fade', str(a.fade),
        '--fit', str(a.fit),
        '--final', str(a.final),
        '--out', str(shrink_out),
    ]
    
    if run_cmd(shrink_cmd, "SHRINK") != 0:
        print("Shrink failed!")
        sys.exit(1)
    
    # Find the shrunk output
    target_file = shrink_out / f'{R}x{a.target_B}x{a.target_C}.npz'
    if not target_file.exists():
        # Try last.npz
        target_file = shrink_out / 'last.npz'
    
    if not target_file.exists():
        print(f"No output found in {shrink_out}")
        sys.exit(1)
    
    print(f"\nShrunk file: {target_file}")
    d = np.load(target_file)
    print(f"Shape: {d['W'].shape}")
    
    if a.skip_pipeline:
        print("Skipping pipeline (--skip-pipeline)")
        return
    
    # Run the finish pipeline
    recipe = HERE / 'recipes' / 'finish_shrunk.toml'
    if not recipe.exists():
        print(f"Recipe not found: {recipe}")
        sys.exit(1)
    
    pipeline_cmd = [
        str(HERE / 'run.py'),
        str(recipe),
        '--start', str(target_file),
        '--out', str(a.out / 'pipeline'),
    ]
    
    if run_cmd(pipeline_cmd, "PIPELINE") != 0:
        print("Pipeline failed!")
        sys.exit(1)
    
    print("\n=== DONE ===")
    # Print results
    result_json = a.out / 'pipeline' / 'result.json'
    if result_json.exists():
        import json
        result = json.loads(result_json.read_text())
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
