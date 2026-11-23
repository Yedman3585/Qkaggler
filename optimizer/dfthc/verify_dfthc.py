"""Evaluate DFTHC submissions as vectors or recovered integer angle words.

Both entry points use Qaggle's parameters directly:

    w_b_rho_c[rho, b, c] : lowercase weights, shape (R, B, C)
    W_rho_c[rho, c]      : identity offsets, shape (R, C)
    S[p, q]             : symmetric particle-number shift, shape (N, N)
    beta_1              : scalar particle-number shift

The vector table has shape (N + R*B, N). Its first N rows specify the
one-body eigenvectors; the remaining rows specify the two-body vectors in
(rho, b) order. Each row is normalized independently, as for the optimizer's
U parameters. The one-body rows need not be mutually orthogonal.

The integer table has shape (N + R*B, N - 1). It contains the recovered raw
angle words obtained by kickmix simulation.
Each word encodes theta/pi in [0, 2), with one integer bit and brot - 1
fractional bits. The caller chooses rounding and runs the circuit simulation.

ref contains only the original FCIDUMP inputs:

    ref = {"oei": oei, "tei": tei, "electrons": num_electrons, "ecore": ecore}

oei has shape (N, N). tei has shape (N, N, N, N), with real chemist-order
integrals tei[p, q, r, s] = (pq|rs). Neither array is symmetry shifted or
rescaled. The verifier constructs the one-body coefficient of E_pq internally:

    H1_THC[p, q] = oei[p, q] - 0.5 * sum_r tei[p, r, r, q].

error_1 compares that coefficient, and error_2 compares the full two-electron
tensor after undoing S. No packed integrals or pair indices are required.
energy_upper is a supplied total reference energy, in the same scalar
convention as ecore. E_gap is estimated from that energy, not obtained by
diagonalizing the full many-body Hamiltonian.

Both paths return error_1, error_2, Lambda, E_SOS, E_gap, lambda_eff, and
gap_in_domain. A candidate outside 0 <= E_gap <= 2*Lambda retains its error
diagnostics but has lambda_eff=NaN. No training penalty or clipping is applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np
import numpy.typing as npt
from dfthc import df_thc_sga

from dfthc.verify_angle_k import (
    build_h1_prime,
    h1_thc_reconstruction_error,
    reconstruct_h2,
    reconstruct_u_vectors,
    reconstruct_x,
    split_u_vectors,
)


def evaluate_dfthc_vectors(
    u_all: npt.ArrayLike,
    w_b_rho_c: npt.ArrayLike,
    W_rho_c: npt.ArrayLike,
    S: npt.ArrayLike,
    beta_1: float,
    ref: Mapping[str, Any],
    *,
    energy_upper: float,
) -> dict[str, float | bool]:
    """Evaluate all N+RB submitted vector rows, without angle quantization.

    Eigenvalues are derived from the target h1_prime and matched internally
    to the submitted one-body rows. beta_1 is already in Qaggle notation;
    no archive-specific trace correction is applied here.
    """
    vectors = np.asarray(u_all, dtype=np.float64)
    w_b_rho_c = np.asarray(w_b_rho_c, dtype=np.float64)
    W_rho_c = np.asarray(W_rho_c, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    oei = np.asarray(ref["oei"], dtype=np.float64)
    tei = np.asarray(ref["tei"], dtype=np.float64)
    if oei.ndim != 2 or oei.shape[0] != oei.shape[1]:
        raise ValueError("ref['oei'] must have shape (N, N)")
    num_orbitals = oei.shape[0]
    if tei.shape != (num_orbitals,) * 4:
        raise ValueError("ref['tei'] must have shape (N, N, N, N) in chemist order")
    if w_b_rho_c.ndim != 3 or min(w_b_rho_c.shape) < 1:
        raise ValueError("w_b_rho_c must have shape (R, B, C) with positive sizes")
    rank_r, bases_b, copies_c = w_b_rho_c.shape
    if vectors.shape != (num_orbitals + rank_r * bases_b, num_orbitals):
        raise ValueError("u_all must have shape (N + R*B, N)")
    if W_rho_c.shape != (rank_r, copies_c):
        raise ValueError("W_rho_c must have shape (R, C)")
    if S.shape != (num_orbitals, num_orbitals) or not np.allclose(S, S.T):
        raise ValueError("S must be a symmetric (N, N) matrix")
    if not all(np.all(np.isfinite(a)) for a in (vectors, w_b_rho_c, W_rho_c, S)):
        raise ValueError("vectors, w, W, and S must contain finite values")
    if not np.isfinite(beta_1) or not np.isfinite(energy_upper):
        raise ValueError("beta_1 and energy_upper must be finite scalars")
    row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        raise ValueError("submitted vector rows must have finite, nonzero norms")
    one_body_vectors, u_b_rho = split_u_vectors(vectors / row_norms, w_b_rho_c)

    X_rho_c = reconstruct_x(u_b_rho, w_b_rho_c)
    # Match the package's symmetric lower-triangle convention despite float32
    # contraction roundoff. Use this same X in the target and both error checks.
    X_rho_c = np.tril(X_rho_c) + np.swapaxes(np.tril(X_rho_c, k=-1), -1, -2)
    # Converting the raw FCIDUMP Hamiltonian to E_pq E_rs contributes this
    # one-body contraction of the original, unshifted chemist ERIs.
    h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
    h1_prime = build_h1_prime(
        h1_thc, S, beta_1, ref["electrons"], W_rho_c, X_rho_c
    )
    eigenvalues = np.linalg.eigvalsh(h1_prime)
    error_1 = h1_thc_reconstruction_error(
        one_body_vectors,
        h1_thc,
        S,
        beta_1,
        ref["electrons"],
        W_rho_c,
        X_rho_c,
    )
    # error_2 = ||sum_rc X_rc tensor X_rc - (S tensor I + I tensor S)/2 - tei||_F.
    # Subtract the identity terms on their diagonal pairs without allocating
    # additional N^4 symmetry-shift tensors or modifying the input integrals.
    h2_delta = reconstruct_h2(X_rho_c)
    h2_delta -= tei
    diagonal = np.arange(num_orbitals)
    h2_delta[:, :, diagonal, diagonal] -= 0.5 * S[:, :, None]
    h2_delta[diagonal, diagonal, :, :] -= 0.5 * S[None, :, :]
    error_2 = float(np.linalg.norm(h2_delta.ravel()))

    # Qaggle Eq. (42): retain the signed generator weights even when the
    # submitted one-body rows are not orthogonal after finite-bit encoding.
    lambda_h1 = float(np.sum(np.abs(eigenvalues)))
    # The shared helper's archive keys store lowercase w and w_B = W + sum_b w.
    lambda_h2 = float(df_thc_sga.Compute_df_thc_sga_lambda_H2({
        "W": w_b_rho_c,
        "H2_I": W_rho_c + np.sum(w_b_rho_c, axis=1),
    }))
    Lambda = lambda_h1 + lambda_h2

    # Eq. (43): every submitted vector has unit norm, so Tr(h1_prime_decoded)
    # equals sum_j lambda_j even if the vectors are not mutually orthogonal.
    E_SOS = float(
        ref["ecore"]
        - ref["electrons"] * beta_1
        - (lambda_h1 - np.sum(eigenvalues))
        - 0.5 * np.sum(W_rho_c**2)
    )
    E_gap = float(energy_upper - E_SOS)
    gap_in_domain = 0.0 <= E_gap <= 2.0 * Lambda
    lambda_eff = (
        float(np.sqrt(E_gap * (2.0 * Lambda - E_gap)))
        if gap_in_domain else float("nan")
    )
    return {
        "error_1": error_1,
        "error_2": error_2,
        "Lambda": Lambda,
        "E_SOS": E_SOS,
        "E_gap": E_gap,
        "lambda_eff": lambda_eff,
        "gap_in_domain": gap_in_domain,
    }


def evaluate_dfthc_angle_words(
    angle_words: npt.ArrayLike,
    w_b_rho_c: npt.ArrayLike,
    W_rho_c: npt.ArrayLike,
    S: npt.ArrayLike,
    beta_1: float,
    ref: Mapping[str, Any],
    *,
    brot: int,
    tree_edges: npt.ArrayLike,
    energy_upper: float,
) -> dict[str, float | bool]:
    """Decode recovered integer words and evaluate the represented vectors.

    angle_words has N+RB rows and N-1 columns. Simulation and rounding belong
    to the caller; this function accepts neither ANGLE_k circuits nor masks.
    tree_edges gives the ordered Givens pairs used to interpret each row.
    """
    words = np.asarray(angle_words)
    oei = np.asarray(ref["oei"])
    if oei.ndim != 2 or oei.shape[0] != oei.shape[1]:
        raise ValueError("ref['oei'] must have shape (N, N)")
    num_orbitals = oei.shape[0]
    try:
        brot = index(brot)
    except TypeError as exc:
        raise ValueError("brot must be an integer between 1 and 63") from exc
    if not 1 <= brot <= 63:
        raise ValueError("brot must be an integer between 1 and 63")
    if words.ndim != 2 or words.shape[1] != num_orbitals - 1:
        raise ValueError("angle_words must have shape (N + R*B, N - 1)")
    if not np.issubdtype(words.dtype, np.integer):
        raise ValueError("angle_words must contain integers")
    if np.any(words < 0) or np.any(words >= 1 << brot):
        raise ValueError("angle_words must lie in [0, 2**brot)")
    edges = np.asarray(tree_edges)
    if edges.shape != (num_orbitals - 1, 2) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("tree_edges must contain N-1 ordered integer vertex pairs")
    if np.any(edges < 0) or np.any(edges >= num_orbitals):
        raise ValueError("tree_edges vertices must lie in [0, N)")
    visited = {0}
    for parent, child in edges:
        if parent not in visited or child in visited:
            raise ValueError("tree_edges must form a tree rooted at 0 in parent-before-child order")
        visited.add(child)

    u_all = reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
    return evaluate_dfthc_vectors(
        u_all, w_b_rho_c, W_rho_c, S, beta_1, ref, energy_upper=energy_upper
    )


__all__ = ["evaluate_dfthc_vectors", "evaluate_dfthc_angle_words"]
