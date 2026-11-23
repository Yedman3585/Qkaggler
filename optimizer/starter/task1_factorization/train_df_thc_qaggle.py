"""Minimal training example for the revised DF-THC objective.

The historical filename refers to an older version of the specification.  In
the current Qaggle draft, Eqs. (32)--(43) add the particle-number symmetry
shift ``(S, beta)``, Eq. (47) defines

    lambda_eff = sqrt(E_gap * (2 * Lambda - E_gap)),

and Eq. (48) minimizes ``lambda_eff`` subject to a 20 mHa Frobenius-error
constraint.  Since Optax's Adam optimizer is unconstrained, this example uses
ReLU penalties for that constraint and for leaving Eq. (45)'s physical gap
interval during early, inaccurate optimization steps.

The objective is implemented once in ``dfthc.df_thc_sga``.  This file only
selects an example system and calls the existing parameter initializer and
training loop from ``dfthc.train_df_thc_sga``.

For FeMoCo76o113e the symmetry shift ``(S, beta)`` is fixed by the
competition verifier: ``Hsym_lower`` and ``H1_I`` are read from the packaged
``dfthcsos/FeMoCo76o113e.fixed_shifts.npz`` and held constant during
training (``frozen_symmetry_params``), whether the run is seeded from a
baseline or initialized randomly.

Competition Part 1 (factorization) starter.  Training may be seeded from the
provided baselines in ``baselines/``, but those files are read-only: every
output is written under ``starter/task1_factorization/runs/`` by default, an
output path inside ``baselines/`` is rejected, and existing files are never
overwritten.  The output path is validated before training starts.
"""

import os
import sys
import tomllib
from typing import Any

import jax.numpy as jnp
import numpy as np

# Development shim: allow running this file directly without installing; the
# repository root holds dfthc/, verifier/ and starter/.
_SOURCE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SOURCE_DIRECTORY not in sys.path:
    sys.path.insert(0, _SOURCE_DIRECTORY)

from dfthc import df_thc_sga  # noqa: E402
from dfthc.train_df_thc_sga import Initialize, Params, Train  # noqa: E402
from starter.task1_factorization.solution_io import (  # noqa: E402
    check_output_path,
    read_factorization,
    save_solution_npz,
)

# data/ and baselines/ live at the repository root, two levels above
# starter/task1_factorization/.
REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
INTEGRALS_DIRECTORY = os.path.join(REPOSITORY_ROOT, 'data', 'integrals')
# Provided baseline factorizations: read-only, used only as seeds.
BASELINES_DIRECTORY = os.path.join(REPOSITORY_ROOT, 'baselines')
# Default location for every solution written by this script.
RUNS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
# The competition verifier's FeMoCo76o113e data, one level above the dfthc-sos
# tree: the reference FCIDUMP and the fixed symmetry shift (H1_I and
# Hsym_lower of the provided baseline).
PACKAGE_DIRECTORY = os.path.dirname(REPOSITORY_ROOT)
PACKAGED_FCIDUMP = os.path.join(PACKAGE_DIRECTORY, 'FeMoCo76o113e.fcidump')
FIXED_SHIFTS_FILE = os.path.join(PACKAGE_DIRECTORY, 'FeMoCo76o113e.fixed_shifts.npz')
# The provided FeMoCo76o113e baseline as an .npz, shipped with the starter.
PACKAGED_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'FeMoco76_65_60_60.npz'
)
# The competition verifier's acceptance floors (keep in sync with
# LAMBDA_FLOOR and E_GAP_FLOOR in dfthcsos/verify.py). Raw Lambda and the
# SOS gap enter the FeMoCo76o113e objective as exact penalties. The verifier
# requires strict > at the floors; a zero penalty alone does not imply acceptance.
FEMOCO_FLOORS = {'lambda_floor': 49.0, 'sos_gap_floor': 3.914898}
# Training settings for FeMoCo76o113e seeded from its baseline. The generic
# defaults (rho=0.001, epsilon_dfthc=0.02) let rho * lambda_eff dominate the
# error penalty and drift past the verifier's 20 mHa limit.
FEMOCO_TRAINING = {'rho': 0.0001, 'epsilon_dfthc': 0.010}


def _load_systems() -> dict[str, dict[str, Any]]:
    """Read the system registry data/systems.toml.

    Without the research data bundle (no data/systems.toml), the registry
    holds only FeMoCo76o113e, with the packaged FCIDUMP and .npz baseline.
    """
    registry_file = os.path.join(REPOSITORY_ROOT, 'data', 'systems.toml')
    if not os.path.isfile(registry_file):
        return {
            'FeMoCo76o113e': {
                'fcidump': PACKAGED_FCIDUMP,
                'best_solution': PACKAGED_BASELINE,
                'upper_energy': -22140.27335,
                'rank': 65,
                'bases': 60,
                'copies': 60,
                **FEMOCO_TRAINING,
            }
        }
    with open(registry_file, 'rb') as fid:
        registry = tomllib.load(fid)['systems']
    return {
        name: {
            'fcidump': entry['fcidump'],
            'best_solution': entry['baseline'],
            'upper_energy': entry['upper_energy'],
            'rank': entry['rank'],
            'bases': entry['bases'],
            'copies': entry['copies'],
            **(FEMOCO_TRAINING if name == 'FeMoCo76o113e' else {}),
        }
        for name, entry in registry.items()
    }


# Example systems with the factorization sizes of their provided baselines in
# baselines/ (see data/systems.toml).
SYSTEMS = _load_systems()


def check_output_file(output_file: str) -> str:
    """Return the absolute output path, refusing baselines/ and existing files.

    Called before training starts and again right before saving, so a long run
    is never lost to a bad path and a baseline is never overwritten.
    """
    output_path = os.path.realpath(output_file)
    baselines = os.path.realpath(BASELINES_DIRECTORY)
    if os.path.commonpath([output_path, baselines]) == baselines:
        raise ValueError(
            f'Refusing to write {output_path}: baselines/ is read-only. '
            f'Write outputs elsewhere, e.g. {RUNS_DIRECTORY}.'
        )
    return str(check_output_path(output_path))


def load_seed_params(
    solution_file: str,
    rank: int | None = None,
    bases: int | None = None,
    copies: int | None = None,
    orbitals: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
    """Load NPZ parameters and diagnostics, adapting to `(rank, bases, copies)`."""
    seed_params, values = read_factorization(solution_file)
    r0, b0, n0 = seed_params['U'].shape
    _, _, c0 = seed_params['W'].shape

    target_r = r0 if rank is None else rank
    target_b = b0 if bases is None else bases
    target_c = c0 if copies is None else copies
    target_n = n0 if orbitals is None else orbitals

    if target_n != n0:
        raise ValueError(
            f'Orbital mismatch between seed solution ({n0}) and target ({target_n}).'
        )

    if (target_r, target_b, target_c) == (r0, b0, c0):
        params = {
            k: jnp.asarray(v, dtype=df_thc_sga.dtype) for k, v in seed_params.items()
        }
    else:
        base = Params('Qaggle', target_r, target_b, target_c, target_n, seed)
        r_min = min(target_r, r0)
        b_min = min(target_b, b0)
        c_min = min(target_c, c0)
        params = {
            'U': base['U'].at[:r_min, :b_min, :].set(
                jnp.asarray(seed_params['U'][:r_min, :b_min, :], dtype=df_thc_sga.dtype)
            ),
            'W': base['W'].at[:r_min, :b_min, :c_min].set(
                jnp.asarray(
                    seed_params['W'][:r_min, :b_min, :c_min], dtype=df_thc_sga.dtype
                )
            ),
            'H2_I': base['H2_I'].at[:r_min, :c_min].set(
                jnp.asarray(seed_params['H2_I'][:r_min, :c_min], dtype=df_thc_sga.dtype)
            ),
            'Hsym_lower': jnp.asarray(seed_params['Hsym_lower'], dtype=df_thc_sga.dtype),
            'H1_I': jnp.asarray(seed_params['H1_I'], dtype=df_thc_sga.dtype),
        }

    return params, values


def frozen_symmetry_params(ref: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """The fixed ``Hsym_lower`` and ``H1_I`` for FeMoCo76o113e; ``{}`` for other systems."""
    if (ref['orbitals'], ref['electrons']) != (76, 113):
        return {}
    with np.load(FIXED_SHIFTS_FILE, allow_pickle=False) as arrays:
        return {
            key: jnp.asarray(arrays[key], dtype=df_thc_sga.dtype)
            for key in ('Hsym_lower', 'H1_I')
        }


def save_solution(
    output_file: str,
    params_best: dict[str, Any],
    values: dict[str, Any],
    ref: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save optimized arrays, diagnostics, and metadata in a numeric NPZ archive."""
    output_path = check_output_file(output_file)
    params_np = {k: np.asarray(v) for k, v in params_best.items()}
    values_out = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v))
        for k, v in values.items()
    }

    # Populate legacy archive aliases expected by read_spinfree_and_get_hamiltonian_delta.py
    if 'error_h2' not in values_out and 'epsilon_fro' in values_out:
        values_out['error_h2'] = values_out['epsilon_fro']
    if 'lambda_DFTHC' not in values_out and 'Lambda' in values_out:
        values_out['lambda_DFTHC'] = values_out['Lambda']
    if 'H0' not in values_out and ref is not None:
        values_out['H0'] = float(
            -df_thc_sga.Compute_H0_shifted_by_Hsym_lower(params_best, ref)
        )

    # Unpack the paper coordinates (S, beta, W_rho_c) from (Hsym_lower, H1_I, H2_I)
    S_matrix = (
        np.asarray(df_thc_sga.H1_from_lower(params_best['Hsym_lower'], ref))
        if ref is not None
        else None
    )
    beta_val = (
        float(params_np['H1_I'] - 0.5 * np.trace(S_matrix))
        if S_matrix is not None
        else None
    )
    W_offset = params_np['H2_I'] - np.sum(params_np['W'], axis=1)
    if beta_val is not None:
        values_out['beta'] = beta_val

    save_solution_npz(
        output_path,
        params_np,
        values=values_out,
        metadata=metadata,
        extra_arrays={'S': S_matrix, 'beta': beta_val, 'W_offset': W_offset},
    )
    print(f'Saved optimized solution to {output_path}')
    return output_path


def train_df_thc(
    fcidump_file: str,
    upper_energy: float,
    rank: int = 14,
    bases: int = 15,
    copies: int = 5,
    epochs: int = 2000,
    lr: float = 0.001,
    epsilon_dfthc: float = 0.020,  # 20 mHa, Eq. (48)
    rho: float = 0.001,
    seed: int = 0,
    print_interval: int = 200,
    seed_solution_file: str | None = None,
    output_file: str | None = None,
):
    """Train with the shared initializer and optimizer; rho weights lambda_eff."""
    if output_file is not None:
        # Fail now rather than after a long run if the output path is invalid.
        check_output_file(output_file)
    ref = Initialize(fcidump_file, upper_energy)
    if seed_solution_file is not None:
        print(f'Seeding parameters from {seed_solution_file}')
        params, _ = load_seed_params(
            seed_solution_file,
            rank=rank,
            bases=bases,
            copies=copies,
            orbitals=ref['orbitals'],
            seed=seed,
        )
    else:
        params = Params('Qaggle', rank, bases, copies, ref['orbitals'], seed)

    frozen_params = frozen_symmetry_params(ref)
    params = {**params, **frozen_params}
    solution = Train(
        generating_set='Qaggle',
        method='DF-THC-SOS',
        epochs=epochs,
        learning_rate=lr,
        rho=rho,
        params=params,
        ref=ref,
        print_interval=print_interval,
        epsilon_dfthc=epsilon_dfthc,
        frozen_params=frozen_params,
        **(FEMOCO_FLOORS if frozen_params else {}),
    )

    if output_file is not None:
        save_solution(
            output_file=output_file,
            params_best=solution['params_best'],
            values=solution['values'],
            ref=ref,
            metadata={
                'fcidump_file': fcidump_file,
                'upper_energy': upper_energy,
                'rank': int(solution['params_best']['U'].shape[0]),
                'bases': int(solution['params_best']['U'].shape[1]),
                'copies': int(solution['params_best']['W'].shape[2]),
                'epochs': epochs,
                'lr': lr,
                'epsilon_dfthc': epsilon_dfthc,
                'rho': rho,
                'seed': seed,
                'seed_solution_file': seed_solution_file,
            },
        )

    return solution['params_best'], solution['values']


def train_system(
    system_name: str,
    epochs: int = 1000,
    lr: float = 0.001,
    rho: float | None = None,
    seed: int = 0,
    print_interval: int = 200,
    rank: int | None = None,
    bases: int | None = None,
    copies: int | None = None,
    pad: int = 0,
    use_best_solution: bool = True,
    seed_solution_file: str | None = None,
    save: bool = True,
    output_file: str | None = None,
    epsilon_dfthc: float | None = None,
):
    """Helper to train any system defined in SYSTEMS.

    ``rho`` and ``epsilon_dfthc`` default to the system's registry entry
    (``FEMOCO_TRAINING`` for FeMoCo76o113e), else 0.001 and 0.02.
    Seeds from the system's baseline in baselines/ by default and saves to
    runs/<system>_<R>_<B>_<C>_ep<epochs>_seed<seed>.npz unless output_file
    is given. Outputs inside baselines/ and existing files are rejected.
    """
    cfg = SYSTEMS[system_name]
    fcidump = os.path.join(INTEGRALS_DIRECTORY, cfg['fcidump'])
    if system_name == 'FeMoCo76o113e' and not os.path.isfile(fcidump):
        # Shipping the complete registry must not require a second copy of
        # the competition FCIDUMP in the optional research-data directory.
        fcidump = PACKAGED_FCIDUMP

    r = (cfg['rank'] + pad) if rank is None else rank
    b = (cfg['bases'] + pad) if bases is None else bases
    c = (cfg['copies'] + pad) if copies is None else copies
    rho = cfg.get('rho', 0.001) if rho is None else rho
    epsilon_dfthc = cfg.get('epsilon_dfthc', 0.02) if epsilon_dfthc is None else epsilon_dfthc

    if seed_solution_file is None and use_best_solution and 'best_solution' in cfg:
        candidate = os.path.join(BASELINES_DIRECTORY, cfg['best_solution'])
        if os.path.isfile(candidate):
            seed_solution_file = candidate
        else:
            print(f'Warning: seed solution not found at {candidate}; initializing randomly.')

    if output_file is None and save:
        output_file = os.path.join(
            RUNS_DIRECTORY,
            f'{system_name}_{r}_{b}_{c}_ep{epochs}_seed{seed}.npz',
        )

    return train_df_thc(
        fcidump_file=fcidump,
        upper_energy=cfg['upper_energy'],
        rank=r,
        bases=b,
        copies=c,
        epochs=epochs,
        lr=lr,
        rho=rho,
        seed=seed,
        print_interval=print_interval,
        seed_solution_file=seed_solution_file,
        output_file=output_file,
        epsilon_dfthc=epsilon_dfthc,
    )


if __name__ == "__main__":
    # Active demonstration run on Fe2S2 seeded from its best_solution:
    # train_system('Fe2S2-20o30e', epochs=50_000, lr=0.001, print_interval=200, rank=15, bases=20, copies=20, pad=0, rho=0.001, epsilon_dfthc=0.01)

    # Other systems from read_spinfree_and_get_hamiltonian_delta.py (uncomment to run).
    # Outputs go to runs/. Seed files not listed in data/systems.toml are
    # earlier intermediate solutions; point seed_solution_file at runs/ for those.
    # train_system('Fe4S4-36o54e', epochs=1000, lr=0.001, print_interval=200)
    # train_system( 'Fe4S4-36o54e', epochs=200_000, lr=0.001, print_interval=200, rank=32, bases=22, copies=22, pad=0, rho=0.001, epsilon_dfthc=0.001,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'Fe4S4-36o54e_30_20_20_ep100000_seed0.npz'))

    # train_system('FeMoCo54o54e', epochs=50_000, lr=0.001, print_interval=200, rank=40, bases=45, copies=45, seed=0, rho=0.001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'FeMoCo54o54e_35_40_40_ep50000_seed0.npz'))
    # train_system('FeMoCo76o113e', epochs=50_000, lr=0.001, print_interval=200, rank=65, bases=60, copies=60, seed=0, rho=0.0001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'FeMoCo76o113e_65_60_60_ep50000_seed0.npz'))

    # train_system('cpd1X58o63e', epochs=60_000, lr=0.001, print_interval=200, rank=40, bases=45, copies=45, seed=0, rho=0.0001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'cpd1X58o63e_40_45_45_ep60000_seed0.npz'))

    # train_system('XVIII56o64e', epochs=60_000, lr=0.001, print_interval=200,  rank=50, bases=50, copies=50, seed=0, rho=0.0001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'XVIII56o64e_45_50_50_ep60000_seed0.npz'))

    # train_system('XVIII100o100e', epochs=60_000, lr=0.001, print_interval=200,  rank=70, bases=70, copies=70, seed=0, rho=0.0001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'XVIII100o100e_60_60_60_ep60000_seed0.npz'))
    # train_system('XVIII150o150e', epochs=50_000, lr=0.001, print_interval=200,  rank=120, bases=120, copies=120, seed=0, rho=0.0001, epsilon_dfthc=0.01,
    # seed_solution_file=os.path.join(BASELINES_DIRECTORY, 'XVIII150o150e_120_120_120_ep75000_seed0.npz'))

    # The competition system, seeded from the packaged baseline with the
    # symmetry shift frozen and the verifier's floors in the objective.
    train_system('FeMoCo76o113e', epochs=1000, lr=0.0001, print_interval=100)
