"""Fast Windows-parallelized runner for Benchmark I (`verify_submission_invsqrt.py`).

Uses all CPU cores on Windows via `spawn` multiprocessing with per-worker initialization
(avoiding the `'fork' not in mp.get_all_start_methods()` single-core fallback in the
official script) and SIMD-friendly default batch sizing, while importing all validation,
reference math, and scoring logic directly from `verify_submission_invsqrt.py`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
import time
from fractions import Fraction

import kickmix as km
import numpy as np

import verify_submission_invsqrt as v

_G_STATE: dict = {}


def _init_worker(circuit_path: str, fmt: str, spec: v.ProblemSpecInvSqrt, batch_size: int) -> None:
    circuit = km.Circuit.from_file(circuit_path, format=fmt)
    x_reg, scratch = v.validate_registers(circuit, spec)
    target, tol = v.reference_values(spec, np.arange(1, spec.max_x + 1, dtype=np.uint64))
    _G_STATE['circuit'] = circuit
    _G_STATE['spec'] = spec
    _G_STATE['x_reg'] = x_reg
    _G_STATE['scratch'] = scratch
    _G_STATE['target'] = target
    _G_STATE['tol'] = tol
    _G_STATE['batch_size'] = batch_size


def _run_chunk(task: tuple[int, int, int]) -> tuple[int, tuple[float, int], int, int, int]:
    chunk_start, chunk_stop, sim_seed = task
    circuit = _G_STATE['circuit']
    spec = _G_STATE['spec']
    x_reg = _G_STATE['x_reg']
    scratch = _G_STATE['scratch']
    target = _G_STATE['target']
    tol = _G_STATE['tol']
    batch_size = _G_STATE['batch_size']

    sim = km.Simulator(
        batch_size=batch_size,
        debug_print=False,
        counting_operations=True,
        seed=sim_seed,
    )
    min_margin = 1 << 64
    worst = (-1.0, 0)
    half_turns_to_units = v.half_turns_to_units
    from_bits = v.from_bits
    to_bits = v.to_bits
    sample_shots = v.sample_shots
    has_scratch = len(scratch) > 0

    for start in range(chunk_start, chunk_stop, batch_size):
        xs = sample_shots(spec, start, start + batch_size)
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, to_bits(xs, spec.width))
        sim.do(circuit)

        if not np.array_equal(from_bits(sim.read_across_shots(x_reg)), xs):
            raise AssertionError('x was not restored.')
        if has_scratch:
            dirty = sim.read_across_shots(scratch).any(axis=1)
            if dirty.any():
                raise AssertionError(
                    f'Scratch qubit {scratch[int(np.argmax(dirty))]} did not return to |0>.')

        theta = np.fromiter(
            (half_turns_to_units(sim.read_shot_phase(i)) for i in range(batch_size)),
            dtype=np.uint64,
            count=batch_size,
        )
        idx = xs - np.uint64(1)
        allowed = tol[idx]
        error = (theta - target[idx]).view(np.int64)
        used = np.abs(error).view(np.uint64)
        bad = np.flatnonzero(used > allowed)
        if len(bad):
            i = int(bad[0])
            raise AssertionError(
                f'x={xs[i]}: phase error {error[i] / 2**64:.3e} turns '
                f'exceeds tolerance {allowed[i] / 2**64:.3e}.'
            )
        ratio = np.divide(used, allowed, out=np.zeros(batch_size), where=allowed > 0)
        min_margin = min(min_margin, int((allowed - used).min()))
        worst = max(worst, (float(ratio.max()), -int(xs[ratio == ratio.max()].min())))

    stats = sim.operation_statistics()
    return (
        min_margin,
        worst,
        int(stats.num_toffoli_like),
        int(stats.num_t_like),
        int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
    )


def pick_batch_size(num_shots: int, requested: int | None) -> int:
    if requested is not None:
        if num_shots % requested != 0:
            raise ValueError(f'num_shots={num_shots} must be a multiple of batch_size={requested}.')
        return requested
    for candidate in (1024, 768, 512, 1344, 256, 192, 128, 64, 32, 16, 8, 4, 2, 1):
        if num_shots % candidate == 0:
            return candidate
    return 1


def fast_verify(circuit_path: str, fmt: str, spec: v.ProblemSpecInvSqrt, *,
                seed: int | None = None, salt: str = '',
                batch_size: int | None = None, max_workers: int | None = None) -> dict:
    circuit = km.Circuit.from_file(circuit_path, format=fmt)
    num_shots = spec.num_shots
    bs = pick_batch_size(num_shots, batch_size)
    circuit_hash = v._circuit_sha256(circuit)
    base_seed = v._initial_seed(circuit_hash, salt) if seed is None else int(seed)

    cpus = os.cpu_count() or 1
    workers = cpus if max_workers is None else max_workers

    # Coarse chunks (~4 per worker) to minimize spawn IPC overhead while keeping load balanced
    total_batches = num_shots // bs
    n_chunks = min(total_batches, max(1, workers * 4))
    batches_per_chunk = max(1, total_batches // n_chunks)
    chunk_shots = batches_per_chunk * bs

    starts = list(range(0, num_shots, chunk_shots))
    seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(
        4 * len(starts), dtype=np.uint64
    ).reshape(-1, 4)
    tasks = [
        (s, min(s + chunk_shots, num_shots), int.from_bytes(seeds[i], 'little'))
        for i, s in enumerate(starts)
    ]

    if workers <= 1 or len(tasks) <= 1:
        _init_worker(circuit_path, fmt, spec, bs)
        results = [_run_chunk(t) for t in tasks]
    else:
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(circuit_path, fmt, spec, bs),
        ) as pool:
            results = list(pool.map(_run_chunk, tasks))

    min_margin = min(r[0] for r in results)
    worst = max(r[1] for r in results)

    toffoli_count = sum(r[2] for r in results) / num_shots
    t_base = sum(r[3] for r in results) / num_shots
    n_rot = sum(r[4] for r in results) / num_shots
    eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
    if n_rot and not eps > 0:
        raise AssertionError('No tolerance left for rotation synthesis error.')
    t_count = t_base + (n_rot * v.rotation_t_count(eps) if n_rot else 0)
    toffolis = toffoli_count + t_count / v.T_PER_TOFFOLI
    return {
        'status': 'PASS',
        'score': circuit.num_qubits * toffolis,
        'toffoli_equivalent': toffolis,
        'toffoli_count': toffoli_count,
        'max_toffoli_count': circuit.operation_statistics().num_toffoli_like,
        't_count': t_base,
        'rotation_count': n_rot,
        'rotation_eps': eps,
        'num_qubits': circuit.num_qubits,
        'reaction_depth': circuit.reaction_depth(),
        'num_ops': len(circuit),
        'verified_shots': num_shots,
        'batch_size': bs,
        'workers': min(workers, len(tasks)),
        'worst_error_over_tolerance': worst[0],
        'worst_x': -worst[1],
        'circuit_sha256': circuit_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('circuit_file')
    parser.add_argument('--format', choices=('auto', 'kmb', 'kmx'), default='auto')
    parser.add_argument('--n-p', type=int, default=v.N_P)
    parser.add_argument('--beta', type=Fraction, default=Fraction(1))
    parser.add_argument('--precision-bits', type=int, default=24)
    parser.add_argument('--shots-per-x', type=int, default=v.DEFAULT_SHOTS_PER_X)
    parser.add_argument('--eta', type=int, default=v.ETA)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--max-x', type=int, default=None)
    parser.add_argument('--system-qubits', type=int, default=None)
    parser.add_argument('--salt', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    args = parser.parse_args()

    n_p = args.n_p
    width = args.width if args.width is not None else 2 * n_p + 2
    max_x = args.max_x if args.max_x is not None else 3 * (2**n_p - 2)**2
    system_qubits = args.system_qubits if args.system_qubits is not None else 3 * args.eta * n_p
    spec = v.ProblemSpecInvSqrt(
        width=width,
        max_x=max_x,
        beta=args.beta,
        precision_bits=args.precision_bits,
        shots_per_x=args.shots_per_x,
        system_qubits=system_qubits,
    )
    t0 = time.perf_counter()
    try:
        summary = fast_verify(
            args.circuit_file, args.format, spec,
            seed=args.seed, salt=args.salt,
            batch_size=args.batch_size, max_workers=args.max_workers,
        )
    except (AssertionError, ValueError, RuntimeError, OSError) as ex:
        print(json.dumps({'status': 'FAIL', 'error': str(ex)}, indent=2))
        sys.exit(1)
    summary['elapsed_sec'] = round(time.perf_counter() - t0, 2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
