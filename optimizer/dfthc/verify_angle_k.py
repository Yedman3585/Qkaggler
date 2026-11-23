"""Verify the streamed ANGLE_k Boolean oracles.

Address and output registers
----------------------------

Let A be the number of vector addresses represented by the ANGLE oracles.

For the complete Qaggle/SOSSA block encoding,

    A = N + R * B,

because the angle table contains N one-body D1/Q1 eigenvectors and R * B
two-body vectors u_b^(rho).

This is the only supported layout. Addresses 0 through N - 1 contain the
one-body vectors. The following R * B addresses contain the two-body vectors
in (rho, b) order.

Each ANGLE_k circuit has:

    address register:
        ceil(log2(A)) physical qubits, representing valid addresses a in [A]

    angle target register:
        J * brot qubits, organized as J angle words of brot bits each

The batch index k is fixed by the identity of the ANGLE_k circuit and is not
a quantum input.

Boolean-oracle verification
---------------------------

For each batch index

    k = 0, ..., K - 1,

where

    K = ceil((N - 1) / J),

and for every valid vector address a in [A], simulate the Qualtran ANGLE_k
Bloq and verify its exact Boolean action

    ANGLE_k:
        |a>|z> -> |a>|z XOR delta[k, a]>,

where

    delta[0, a] = batch[0, a],

and, for k > 0,

    delta[k, a] = batch[k, a] XOR batch[k - 1, a].

The Boolean action is tested by invoking the decomposed Qualtran circuit and
checking its output bit-for-bit. There is intentionally no high-level
``on_classical_vals`` simulation path.

The output target is a THRU register. Therefore the circuit should also be
tested on representative nonzero target values z, rather than only z = 0.

Streaming verification
----------------------

To recover the actual angle words, do not simulate each ANGLE_k independently
with z initialized to zero. For k > 0 that would return an XOR transition
mask rather than the raw angle batch.

Instead, for every valid vector address a, initialize one reusable target
register to zero and invoke all ANGLE oracles sequentially:

    z = 0

    for k in range(K):
        z = ANGLE_k(a, z)

        assert z == batch[k, a]

        read the J resident brot-bit angle words from z

The register history must be

    0
      -> batch[0, a]
      -> batch[1, a]
      -> ...
      -> batch[K - 1, a].

The last batch is zero padded when J does not divide N - 1. Only the first
N - 1 recovered words are interpreted as Givens angles.

The reverse traversal must also be tested:

    for k in reversed(range(K)):
        z = ANGLE_k†(a, z)

After the final ANGLE_0 adjoint call, the target must be exactly zero. This
verifies both the XOR-difference construction and the adjoint traversal. The
underlying XOR QROM is self-inverse, so its Boolean values are unchanged by
the adjoint.

Bit decoding
------------

The QROM and this verifier treat each recovered angle word as an opaque
integer. The caller owns the encoding and rounding policy. For Qaggle's
unsigned convention, one possible decoding uses one integer bit and brot - 1
fractional bits for theta / pi. The same raw word can be viewed as theta /
(2 * pi) with brot fractional bits by the phase-gradient rotation.

After recovering the raw words, the caller decodes each address a and ordered
tree edge t:

    angle_word[a, t] -> theta_over_pi[a, t]

and set

    theta[a, t] = pi * theta_over_pi[a, t].

Do not numerically decode delta[k, a] directly. For k > 0 it is an XOR mask
and does not by itself represent an angle.

Reconstruction of unit vectors
------------------------------

For each vector address a, reconstruct the corresponding N-dimensional unit
vector using the supplied rooted, child-ordered Givens tree.

Initialize

    u_a = (1, 0, ..., 0).

For each ordered tree edge

    e_t = (p_t -> q_t),

apply the planar rotation

    new_u[p_t] = cos(theta[a, t]) * u[p_t]
                 - sin(theta[a, t]) * u[q_t]

    new_u[q_t] = sin(theta[a, t]) * u[p_t]
                 + cos(theta[a, t]) * u[q_t].

All other entries are unchanged.

After processing the N - 1 edges, verify

    ||u_a||_2 = 1

within numerical tolerance.

Reconstruct all N + R * B vectors. Keep the first N rows as the one-body
vectors. For reconstruction of the two-electron tensor, reshape the following
R * B rows into u_b^(rho).

The factorization variables follow Qaggle Eq. (36):

    Theta = {u_b^(rho), w_b^(rho c), W^(rho c)}.

Here lowercase w_b^(rho c) multiplies u_b^(rho) in X^(rho c), while uppercase
W^(rho c) is the separately supplied identity offset.

Reconstruction of the one-electron tensor
-----------------------------------------

First use the final R * B vectors recovered from the QROM bits to construct
``X``. The participant also supplies the symmetry-shift matrix ``S``, scalar
``beta_1``, and identity-offset weights W^(rho c). The exact one-body target in
the E_pq E_rs convention is

    H1_THC[p,q] = H1_core[p,q]
                   - 0.5 * sum_r H2_full[p,r,r,q],

and the ideal matrix whose eigenvectors the first N QROM rows encode is

    h1_prime_ideal[p,q] = H1_THC[p,q]
                           - (eta / 2) * S[p,q]
                           + beta_1 * I[p,q]
                           - sum_(rho,c) W[rho,c] * X[rho,c,p,q].

The user supplies only the eigenvectors, not their eigenvalues. Diagonalize
``h1_prime_ideal`` to obtain the authoritative signed eigenvalues and ideal
eigenvectors. The first N QROM vectors may use a different ordering or global
signs. Match them to the ideal eigenvectors with a one-to-one assignment that
maximizes absolute overlap, then attach the matched eigenvalue to each QROM
vector and reconstruct

    h1_prime_qrom[p,q] = sum_j lambda_matched[j] * u_j[p] * u_j[q].

Finally, undo every term added to H1_THC in the definition of h1_prime:

    h1_reconstructed[p,q] = h1_prime_qrom[p,q]
                              + (eta / 2) * S[p,q]
                              - beta_1 * I[p,q]
                              + sum_(rho,c) W[rho,c] * X[rho,c,p,q].

The one-electron verification is the matrix-level error

    epsilon_1 = ||h1_reconstructed - H1_THC||_F.

This is a post-QROM loss: it depends on the finite-bit vectors actually
decoded from the first N angle-table rows. It is not available to the
continuous DFTHC training objective, which contains no angle words or decoded
QROM vectors.

This comparison is insensitive to global signs and to harmless permutations.
For an exactly degenerate eigenspace, every matched eigenvalue is identical,
so the assignment within that subspace is also immaterial.

Reconstruction of the implemented two-electron tensor
-----------------------------------------------------

Using the reconstructed two-electron vectors and the supplied coefficients
w_b^(rho c), form

    X^(rho c)[p,q]
        = sum_b w_b^(rho c) * u_b^(rho)[p] * u_b^(rho)[q].

Then reconstruct the two-electron integral tensor implemented by the emitted
angle bits:

    h2_tilde[p,q,s,t]
        = sum_(rho,c) X^(rho c)[p,q] * X^(rho c)[s,t].

Equivalently,

    h2_tilde
        = sum_(rho,c) X^(rho c) outer X^(rho c).

The identity offsets W^(rho c), and hence the derived coefficients
w_B^(rho c) = W^(rho c) + sum_b w_b^(rho c), do not enter this two-electron
tensor reconstruction. They contribute to the one-body and constant parts of
the full SOS representation.

Hamiltonian-quality verification
--------------------------------

For an archived spin-free factorization, first undo the particle-number
symmetry shift. In packed-pair form, subtract one half of ``Hsym_lower`` 
from both diagonal pair axes before comparing with the supplied 
two-electron integrals.

``h2_reconstruction_error`` accepts the reconstructed X factors and the full matrix
S, and uses the repository's symmetry-shift residual and Frobenius-norm
helpers to compare against the reference two-electron integrals. Pass the
factors reconstructed from ANGLE_k output to include angle-rounding error.

``compute_sos_metrics`` reuses that error calculation and evaluates the
Qaggle one-body matrix h1_prime of Eq. (38) from the same X and W, giving
lambda_H1, lambda_H2, Lambda, E_SOS, and the SOS gap with the definitions of
``df_thc_sga.Compute_qaggle_df_thc_cost_function``.

The two-electron error is the full-tensor Frobenius norm

    epsilon_fro = ||h2 - h2_tilde||_F.

The implementation passes the DFTHC accuracy requirement when

    epsilon_fro < 20 mHa
                = 0.020 Ha.

This comparison must use the same tensor index convention as the supplied
integrals. No additional factor of 1/2 is included in h2_tilde; the factor
1/2 appears only when the tensor is inserted into the Hamiltonian

    H2 = (1/2) sum_pqrs h2_tilde[p,q,r,s] E_pq E_rs.

Changing the angle batch size J must not change the reconstructed angle
words, reconstructed vectors, or h2_tilde. Any J dependence indicates an
error in batching, zero padding, XOR updates, fixed-point decoding, angle
ordering, or vector-address mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from dfthc import df_thc_sga
from qualtran import Bloq
from scipy.optimize import linear_sum_assignment

from .build_angle_k import AngleK


IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def simulate_angle_k(
    angle_k: Bloq,
    *,
    a: int,
    z: npt.ArrayLike,
) -> tuple[int, IntArray]:
    """Simulate one ANGLE_k, or its adjoint, through its decomposition."""
    address_out, words_out = angle_k.decompose_bloq().call_classically(a=a, z=z)
    return int(address_out), np.asarray(words_out, dtype=np.int64)


def recover_angle_words(
    angle_ks: Sequence[AngleK],
    *,
    num_angles: int,
) -> IntArray:
    """Stream every address forward, then clean the register with adjoints."""
    if not angle_ks:
        raise ValueError("angle_ks must not be empty")

    num_vectors, batch_size = angle_ks[0].qrom.data[0].shape
    # We can look at entry 0 because every QROM is indexed by all N + RB vectors
    # and outputs batch_size angles.
    num_batches = ceil(num_angles / batch_size)
    if len(angle_ks) != num_batches:
        raise ValueError(f"expected {num_batches} ANGLE_k Bloqs, got {len(angle_ks)}")

    forward = tuple(angle_k.decompose_bloq() for angle_k in angle_ks)
    reverse = tuple(angle_k.adjoint().decompose_bloq() for angle_k in angle_ks)
    recovered = np.empty((num_vectors, num_angles), dtype=np.int64)

    for address in range(num_vectors):
        z = np.zeros(batch_size, dtype=np.int64)

        for k, circuit in enumerate(forward):
            address_out, z_out = circuit.call_classically(a=address, z=z)
            if int(address_out) != address:
                raise AssertionError(f"ANGLE_{k} changed the address register")
            z = np.asarray(z_out, dtype=np.int64)

            start = k * batch_size
            stop = min(start + batch_size, num_angles)
            # setting stop-start lets us handle the padded batchsize in the last AngleK
            recovered[address, start:stop] = z[: stop - start]
            if stop - start < batch_size and np.any(z[stop - start :]):
                raise AssertionError("the final ANGLE_k batch is not zero padded")

        for k in reversed(range(num_batches)):
            address_out, z_out = reverse[k].call_classically(a=address, z=z)
            if int(address_out) != address:
                raise AssertionError(f"Adj(ANGLE_{k}) changed the address register")
            z = np.asarray(z_out, dtype=np.int64)

        if np.any(z):
            raise AssertionError("the adjoint ANGLE_k traversal did not clean z")

    return recovered


def reconstruct_u_vectors(
    theta_over_pi: npt.ArrayLike,
    tree_edges: npt.ArrayLike,
) -> FloatArray:
    """Apply the ordered Givens-tree rotations to e_0 for every vector."""
    angles = np.asarray(theta_over_pi, dtype=np.float64)
    edges = np.asarray(tree_edges, dtype=np.int64)
    num_vectors, num_angles = angles.shape
    if edges.shape != (num_angles, 2):
        raise ValueError("tree_edges must have shape (N - 1, 2)")

    vectors = np.zeros((num_vectors, num_angles + 1), dtype=np.float64)
    vectors[:, 0] = 1.0
    for angle_index, (parent, child) in enumerate(edges):
        parent_value = vectors[:, parent].copy()
        child_value = vectors[:, child].copy()
        theta = np.pi * angles[:, angle_index]
        cosine = np.cos(theta)
        sine = np.sin(theta)
        vectors[:, parent] = cosine * parent_value - sine * child_value
        vectors[:, child] = sine * parent_value + cosine * child_value
    return vectors


def split_u_vectors(
    u_all: npt.ArrayLike,
    w_b_rho_c: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Split the required N one-body and following RB two-body vectors."""
    vectors = np.asarray(u_all, dtype=np.float64)
    w_b_rho_c = np.asarray(w_b_rho_c, dtype=np.float64)
    num_vectors, num_orbitals = vectors.shape
    rank_r, bases_b, _ = w_b_rho_c.shape
    expected_num_vectors = num_orbitals + rank_r * bases_b
    if num_vectors != expected_num_vectors:
        raise ValueError(
            f"expected A=N+R*B={expected_num_vectors} vectors, got A={num_vectors}"
        )

    return (
        vectors[:num_orbitals],
        vectors[num_orbitals:].reshape(rank_r, bases_b, num_orbitals),
    )


def reconstruct_x(
    u_b_rho: npt.ArrayLike,
    w_b_rho_c: npt.ArrayLike,
) -> FloatArray:
    """Build X^(rho c) from Qaggle's u_b^(rho) and w_b^(rho c)."""
    return np.asarray(
        df_thc_sga.Compute_H2_sqrt(u_b_rho, w_b_rho_c),
        dtype=np.float64,
    )


def build_h1_prime(
    h1_thc: npt.ArrayLike,
    S: npt.ArrayLike,
    beta_1: float,
    num_electrons: int,
    W_rho_c: npt.ArrayLike,
    X_rho_c: npt.ArrayLike,
) -> FloatArray:
    """Build the ideal shifted one-body matrix whose eigenvectors QROM stores."""
    h1_thc = np.asarray(h1_thc, dtype=np.float64)
    return (
        h1_thc
        - 0.5 * num_electrons * np.asarray(S, dtype=np.float64)
        + beta_1 * np.eye(h1_thc.shape[0])
        - np.einsum("rc,rcpq->pq", W_rho_c, X_rho_c, optimize=True)
    )


def reconstruct_h1_thc(
    qrom_h1_prime_eigenvectors: npt.ArrayLike,
    h1_thc: npt.ArrayLike,
    S: npt.ArrayLike,
    beta_1: float,
    num_electrons: int,
    W_rho_c: npt.ArrayLike,
    X_rho_c: npt.ArrayLike,
) -> FloatArray:
    """Recover H1_THC using QROM eigenvectors and internally derived eigenvalues."""
    qrom_eigenvectors = np.asarray(
        qrom_h1_prime_eigenvectors, dtype=np.float64
    )
    ideal_h1_prime = build_h1_prime(
        h1_thc, S, beta_1, num_electrons, W_rho_c, X_rho_c
    )
    ideal_eigenvalues, ideal_eigenvector_columns = np.linalg.eigh(ideal_h1_prime)

    # A user may permute the vectors or flip their signs. Associate each QROM
    # row with the ideal eigenvector having the best global one-to-one overlap.
    overlaps = np.abs(qrom_eigenvectors @ ideal_eigenvector_columns)
    qrom_indices, ideal_indices = linear_sum_assignment(-overlaps)
    matched_eigenvalues = np.empty_like(ideal_eigenvalues)
    matched_eigenvalues[qrom_indices] = ideal_eigenvalues[ideal_indices]

    h1_prime_qrom = np.einsum(
        "j,jp,jq->pq",
        matched_eigenvalues,
        qrom_eigenvectors,
        qrom_eigenvectors,
        optimize=True,
    )

    # Invert h1_prime = H1_THC - eta*S/2 + beta_1*I - sum_(rho,c) W*X.
    return (
        h1_prime_qrom
        + 0.5 * num_electrons * np.asarray(S, dtype=np.float64)
        - beta_1 * np.eye(ideal_h1_prime.shape[0])
        + np.einsum("rc,rcpq->pq", W_rho_c, X_rho_c, optimize=True)
    )


def h1_thc_reconstruction_error(
    qrom_h1_prime_eigenvectors: npt.ArrayLike,
    h1_thc: npt.ArrayLike,
    S: npt.ArrayLike,
    beta_1: float,
    num_electrons: int,
    W_rho_c: npt.ArrayLike,
    X_rho_c: npt.ArrayLike,
) -> float:
    """Return epsilon_h1 = ||H1_THC(QROM) - H1_THC||_F."""
    reconstructed = reconstruct_h1_thc(
        qrom_h1_prime_eigenvectors,
        h1_thc,
        S,
        beta_1,
        num_electrons,
        W_rho_c,
        X_rho_c,
    )
    return float(np.linalg.norm(reconstructed - np.asarray(h1_thc)))


def reconstruct_h2(X_rho_c: npt.ArrayLike) -> FloatArray:
    """Materialize the bare Gram tensor, before undoing ``Hsym_lower``."""
    factors = np.asarray(X_rho_c, dtype=np.float64)
    return np.einsum("rcpq,rcst->pqst", factors, factors, optimize=True)


def h2_reconstruction_error(
    X_rho_c: npt.ArrayLike,
    S: npt.ArrayLike,
    ref: Mapping[str, Any],
) -> float:
    """Return ||sum_rc X_rc tensor X_rc - (S tensor I + I tensor S)/2 - h2||_F.

    S is the full symmetry-shift matrix; ref contains the original FCIDUMP
    two-electron integrals in packed-pair form. The shared norm restores the
    full four-index multiplicities, using the DFTHC package's numerical
    precision. No angle encoding or rounding is applied.
    """
    S = jnp.asarray(S, dtype=df_thc_sga.dtype)
    shift_params = {"Hsym_lower": S[ref["tril"][0], ref["tril"][1]]}
    h2_delta = df_thc_sga.Compute_H2_residual_after_symmetry_shift(
        shift_params, ref, jnp.asarray(X_rho_c, dtype=df_thc_sga.dtype)
    )
    return float(df_thc_sga.Compute_H2_norm(h2_delta, ref["tril_diag"]))


def compute_sos_metrics(
    X_rho_c: npt.ArrayLike,
    params: Mapping[str, Any],
    ref: Mapping[str, Any],
    *,
    energy_upper: float,
) -> dict[str, float]:
    """Compute Qaggle's epsilon_fro, lambdas, E_SOS, and SOS gap from X.

    Mirrors ``df_thc_sga.Compute_qaggle_df_thc_cost_function`` (Eqs. (38),
    (40)--(45)) but takes ``X_rho_c`` explicitly, so factors reconstructed
    from ANGLE_k output can be passed. The one-body matrix is Eq. (38)'s
    h1_prime, built from the factorization's X and W; it does not use the
    legacy ``Compute_H0_shifted_by_Hsym_lower``, which substitutes the exact
    FCIDUMP h2 for h2_tilde and agrees with Eqs. (43)--(45) only at zero
    two-electron residual.
    """
    X_rho_c = jnp.asarray(X_rho_c, dtype=df_thc_sga.dtype)
    S = df_thc_sga.H1_from_lower(params["Hsym_lower"], ref)
    epsilon_fro = h2_reconstruction_error(X_rho_c, S, ref)

    # Archive coordinates -> paper notation: W = w_B - sum_b w_b and
    # beta = H1_I - Tr(S)/2 (see Compute_qaggle_df_thc_cost_function).
    W_rho_c = params["H2_I"] - jnp.sum(params["W"], axis=1)
    beta = params["H1_I"] - 0.5 * jnp.trace(S)

    # Eq. (38).
    h1_prime = (
        ref["H1_THC"]
        - 0.5 * ref["electrons"] * S
        + beta * jnp.eye(ref["orbitals"], dtype=df_thc_sga.dtype)
        - jnp.einsum("rc,rcpq->pq", W_rho_c, X_rho_c)
    )

    # Eqs. (41)--(45).
    lambda_h1 = df_thc_sga.Compute_H1_lambda(h1_prime)
    lambda_h2 = df_thc_sga.Compute_df_thc_sga_lambda_H2(params)
    Lambda = lambda_h1 + lambda_h2
    E_SOS = (
        ref["ecore"]
        - ref["electrons"] * beta
        - (lambda_h1 - jnp.trace(h1_prime))
        - 0.5 * jnp.sum(jnp.square(W_rho_c))
    )
    sos_gap = energy_upper - E_SOS

    return {
        "epsilon_fro": float(epsilon_fro),
        "lambda_H1": float(lambda_h1),
        "lambda_H2": float(lambda_h2),
        "Lambda": float(Lambda),
        "E_SOS": float(E_SOS),
        "sos_gap": float(sos_gap),
    }


__all__ = [
    "build_h1_prime",
    "compute_sos_metrics",
    "h1_thc_reconstruction_error",
    "h2_reconstruction_error",
    "recover_angle_words",
    "reconstruct_h1_thc",
    "reconstruct_h2",
    "reconstruct_x",
    "reconstruct_u_vectors",
    "simulate_angle_k",
    "split_u_vectors",
]
