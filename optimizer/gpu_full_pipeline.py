"""Full GPU pipeline for dfthcsos optimization.

Two strategies run in parallel on Kaggle's two T4 GPUs:
  GPU 0: Shrink ahoirg R=9 from 143x42 -> 80x42
  GPU 1: df_init R=15 -> merge to R=5 -> shrink to 100x30
"""
import glob
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def find_input():
    """Find the pre-fitted ahoirg 9x143x42 NPZ."""
    # Check local inputs first
    for name in ['ahoirg_9x143x42.npz', '9x143x42.npz']:
        p = HERE / 'inputs' / name
        if p.exists():
            return p
    # Search Kaggle datasets - look for dfthcsos__submission.npz (the ahoirg factorization)
    for p in sorted(Path('/kaggle/input').rglob('*.npz')):
        try:
            d = np.load(p)
            if 'W' in d and d['W'].shape == (9, 143, 42):
                print(f"[find_input] MATCH: {p} W.shape={d['W'].shape} keys={list(d.keys())}", flush=True)
                return p
        except:
            pass
    return None

def run(cmd, label, env_extra=None):
    import os
    env = dict(os.environ, **(env_extra or {}))
    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, '-u', *cmd], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(f"[{label}] {line}", end="", flush=True)
    code = p.wait()
    dt = time.time() - t0
    print(f"[{label}] {'OK' if code == 0 else 'FAILED'} in {dt:.0f}s", flush=True)
    return code

def strategy_shrink():
    """GPU 0: Shrink ahoirg to R=9,B=80,C=42."""
    out = Path('/kaggle/working/strat1_shrink')
    out.mkdir(parents=True, exist_ok=True)

    inp = find_input()
    if inp is None:
        print("[SHRINK-R9] No 9x143x42 input found!", flush=True)
        for p in sorted(Path('/kaggle/input').rglob('*')):
            if p.is_file():
                print(f"  available: {p} ({p.stat().st_size})", flush=True)
        return 1

    # If the NPZ has angle_words/tree_edges (full submission), strip to U/W/H2_I only
    d = dict(np.load(inp))
    if 'angle_words' in d:
        print(f"[SHRINK-R9] Stripping angle_words/tree_edges from input", flush=True)
        stripped = out / 'input_stripped.npz'
        np.savez(stripped, U=d['U'], W=d['W'], H2_I=d['H2_I'])
        inp = stripped

    print(f"[SHRINK-R9] Using: {inp}", flush=True)

    # shrink.py: positional args = seed outdir schedule
    shrink_out = out / 'shrink'
    code = run([
        str(HERE / 'experiments' / 'shrink.py'),
        str(inp),
        str(shrink_out),
        '130x42,120x42,110x42,100x42,90x42,80x42',
        '--merged',
        '--kappa', '5e-5',
        '--fade', '500',
        '--fit', '3000',
        '--final', '8000',
        '--save', '100x42,90x42,80x42',
        '--f32',
    ], "SHRINK-R9", {"CUDA_VISIBLE_DEVICES": "0"})

    if code != 0:
        return code

    for target in ['80x42', '90x42', '100x42']:
        npz = shrink_out / f'9x{target}.npz'
        if not npz.exists():
            continue
        pipe_out = out / f'pipeline_{target}'
        run([
            str(HERE / 'run.py'),
            str(HERE / 'recipes' / 'finish_shrunk.toml'),
            '--start', str(npz),
            '--out', str(pipe_out),
        ], f"PIPE-{target}", {"CUDA_VISIBLE_DEVICES": "0"})
    return 0

def strategy_df_init():
    """GPU 1: df_init R=15 -> merge to R=5 -> shrink."""
    out = Path('/kaggle/working/strat2_dfinit')
    out.mkdir(parents=True, exist_ok=True)

    seed = out / 'seed_15x80x30.npz'
    if not seed.exists():
        # df_init.py: positional args = out R B C
        code = run([
            str(HERE / 'experiments' / 'df_init.py'),
            str(seed), '15', '80', '30',
            '--fit', '5000',
        ], "DFINIT", {"CUDA_VISIBLE_DEVICES": "1"})
        if code != 0:
            return code

    # shrink.py: positional args = seed outdir schedule
    shrink_out = out / 'shrink'
    code = run([
        str(HERE / 'experiments' / 'shrink.py'),
        str(seed),
        str(shrink_out),
        '240x90,200x75,170x60,150x50,130x45,120x40,110x35,100x30',
        '--merge', '3',
        '--kappa', '5e-5',
        '--fade', '500',
        '--fit', '2000',
        '--final', '6000',
        '--save', '120x40,110x35,100x30',
        '--f32',
    ], "SHRINK-R5", {"CUDA_VISIBLE_DEVICES": "1"})

    if code != 0:
        return code

    for target in ['100x30', '110x35', '120x40']:
        npz = shrink_out / f'5x{target}.npz'
        if not npz.exists():
            continue
        pipe_out = out / f'pipeline_{target}'
        run([
            str(HERE / 'run.py'),
            str(HERE / 'recipes' / 'finish_shrunk.toml'),
            '--start', str(npz),
            '--out', str(pipe_out),
        ], f"PIPE-{target}", {"CUDA_VISIBLE_DEVICES": "1"})
    return 0

if __name__ == '__main__':
    import concurrent.futures, json
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        f1 = pool.submit(strategy_shrink)
        f2 = pool.submit(strategy_df_init)
        r1, r2 = f1.result(), f2.result()
        print(f"\nStrategy 1 (shrink R=9): {'OK' if r1==0 else 'FAILED'}")
        print(f"Strategy 2 (df_init R=5): {'OK' if r2==0 else 'FAILED'}")

        for jf in sorted(glob.glob('/kaggle/working/*/pipeline_*/result.json')):
            print(f"\n{jf}:")
            print(json.dumps(json.loads(Path(jf).read_text()), indent=2))
        
        # Also list all NPZ outputs
        print("\n=== Output files ===")
        for p in sorted(Path('/kaggle/working').rglob('*.npz')):
            try:
                d = np.load(p)
                shape = d['W'].shape if 'W' in d else 'no W'
                print(f"  {p}: {shape}")
            except:
                print(f"  {p}: (can't read)")
