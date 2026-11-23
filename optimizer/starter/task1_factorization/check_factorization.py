"""Part 1 (factorization): check a factorization before moving on.

Evaluates the continuous factorization (no angle rounding) with the
competition verifier and reports the accuracy gates and the lambda_eff factor
of the score: the two-body error against the 20 mHa bound, Lambda, the SOS
gap and lambda_eff. Accepts only the verifier's ``.npz`` format.
From the repository root:

    python starter/task1_factorization/check_factorization.py Fe2S2-20o30e
    python starter/task1_factorization/check_factorization.py Fe2S2-20o30e \\
        starter/task1_factorization/runs/my_run.npz

This is kept separate from train_df_thc_qaggle.py so that checking a
factorization does not need the trainer's JAX and Optax.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

# Development shim: allow running this file directly without installing; the
# repository root holds dfthc/, verifier/ and starter/.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from starter.task1_factorization.solution_io import read_factorization  # noqa: E402
from verifier.score import (  # noqa: E402
    ERROR_2_THRESHOLD,
    evaluate_inputs,
    load_inputs,
    score_from_metrics,
)


def check_factorization(
    system: str,
    factorization: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Verify a factorization (``.npz`` only) for ``system``.

    The returned dictionary has the verifier metrics (``error_1``,
    ``error_2``, ``Lambda``, ``E_SOS``, ``E_gap``, ``lambda_eff``,
    ``gap_in_domain``), the factorization sizes ``R``, ``B``, ``C``, ``N``,
    ``passes_error_2`` and ``score`` (the ``lambda_eff`` factor of the score,
    ``inf`` if rejected).
    """
    params, saved_values = read_factorization(factorization)
    inputs = load_inputs(system, params)
    metrics = evaluate_inputs(inputs)
    rank_r, bases_b, copies_c = inputs.w.shape
    result: dict[str, Any] = {
        **metrics,
        "R": rank_r,
        "B": bases_b,
        "C": copies_c,
        "N": inputs.num_orbitals,
        "passes_error_2": bool(metrics["error_2"] <= ERROR_2_THRESHOLD),
        "score": score_from_metrics(metrics),
    }
    if verbose:
        saved = saved_values.get("error_h2", saved_values.get("epsilon_fro"))
        saved_text = (
            "" if saved is None else f"   (training value {1e3 * float(saved):.4f} mHa)"
        )
        verdict = "ok" if result["passes_error_2"] else "FAILS"
        print(f"{system}: {Path(factorization).name}")
        print(f"  R, B, C, N    = {rank_r}, {bases_b}, {copies_c}, {inputs.num_orbitals}")
        print(
            f"  error_2       = {1e3 * metrics['error_2']:.4f} mHa  "
            f"[{verdict}: bound {1e3 * ERROR_2_THRESHOLD:.0f} mHa]{saved_text}"
        )
        print(f"  error_1       = {metrics['error_1']:.3e} Ha (continuous: ~0)")
        print(f"  Lambda        = {metrics['Lambda']:.6f} Ha")
        print(f"  E_SOS         = {metrics['E_SOS']:.6f} Ha")
        print(
            f"  E_gap         = {metrics['E_gap']:.6f} Ha  "
            f"[{'in' if metrics['gap_in_domain'] else 'OUT OF'} domain 0..2*Lambda]"
        )
        print(f"  lambda_eff    = {metrics['lambda_eff']:.6f} Ha")
        print(
            f"  score         = {result['score']} "
            "(lambda_eff factor, or inf; circuits multiply by Toffoli x qubits)"
        )
    return result


__all__ = ["check_factorization"]


if __name__ == "__main__":
    import argparse

    from starter.task1_factorization.solution_io import BASELINES_DIRECTORY
    from verifier.reference import load_systems

    systems = load_systems()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("system", choices=sorted(systems))
    parser.add_argument(
        "factorization", nargs="?", type=Path,
        help="factorization .npz (default: the system's baseline)",
    )
    args = parser.parse_args()
    path = args.factorization or BASELINES_DIRECTORY / systems[args.system]["baseline"]
    result = check_factorization(args.system, path)
    sys.exit(0 if np.isfinite(result["score"]) else 1)
