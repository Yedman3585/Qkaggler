"""Collect DFTHC Givens data and reconstruct the two-electron loss.

The archived spin-free factorizations contain

    U[r, b, p]       with shape (R, B, N),
    W[r, b, c]       with shape (R, B, C).

The stored U vectors are not normalized.  With

    u[r, b, :] = U[r, b, :] / ||U[r, b, :]||_2,

the matrices called ``H2_sqrt`` in this repository are

    H2_sqrt[r, c, p, q]
        = sum_b W[r, b, c] u[r, b, p] u[r, b, q].

Their pair-space Gram matrix reconstructs the two-electron tensor.  The local
factorizations also use a particle-number symmetry shift; undoing that shift
before comparison reproduces ``values["error_h2"]`` from the NPZ archive.

Each u vector is decomposed separately with
``openfermion.givens_decomposition``.  The raw OpenFermion layers and diagonal
are retained. A second, dense two-body angle-table section is provided in
Qaggle's path order. It contains the ``R * B`` rows that follow the ``N``
one-body rows in the complete ANGLE table. For Qaggle's coefficient recurrence
the angle is ``-theta_openfermion`` because the two conventions use opposite
similarity-transform orientations. Its magnitude is the OpenFermion plane
angle, not half that angle. Qaggle angles are stored modulo ``2 * pi`` in
``[0, 2 * pi)``; the corresponding fixed-point input is
``(theta / pi) mod 2`` in ``[0, 2)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import openfermion as of

# H2_sqrt and the two-electron loss come from the dfthc JAX code; keep float64.
jax.config.update("jax_enable_x64", True)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from dfthc import df_thc_sga, load_integrals  # noqa: E402
from starter.task1_factorization.solution_io import read_factorization  # noqa: E402

INTEGRALS_DIRECTORY = REPOSITORY_ROOT / "data" / "integrals"
# Provided baseline factorizations (read-only). The name is kept for the
# existing imports in the tests.
OPTIMIZED_SOLUTIONS_DIRECTORY = REPOSITORY_ROOT / "baselines"
SYSTEMS_REGISTRY_FILE = REPOSITORY_ROOT / "data" / "systems.toml"
GIVENS_ATOL = 1.0e-8
H2_SQRT_ATOL = 1.0e-7

GivensRotation = tuple[int, int, float, float]
GivensLayer = tuple[GivensRotation, ...]


# Per-system data lives in data/systems.toml; see that file for the fields.
with open(SYSTEMS_REGISTRY_FILE, "rb") as _registry_file:
    SYSTEMS_REGISTRY: dict[str, dict[str, Any]] = tomllib.load(_registry_file)[
        "systems"
    ]

# Baseline factorization NPZ (under OPTIMIZED_SOLUTIONS_DIRECTORY) per system.
OPTIMIZED_SOLUTIONS = {
    name: entry["baseline"] for name, entry in SYSTEMS_REGISTRY.items()
}

# FCIDUMP file (under INTEGRALS_DIRECTORY) per system.
FCIDUMP_FILES = {
    name: entry["fcidump"] for name, entry in SYSTEMS_REGISTRY.items()
}

# Upper bound E_upper on the ground-state energy (Hartree) per system.
UPPER_ENERGIES = {
    name: entry["upper_energy"] for name, entry in SYSTEMS_REGISTRY.items()
}


@dataclass(frozen=True)
class GivensVectorRecord:
    """OpenFermion data for one normalized factorization vector."""

    rank_index: int
    basis_index: int
    source_norm: float
    vector: np.ndarray
    layers: tuple[GivensLayer, ...]
    left_unitary: np.ndarray
    diagonal: complex
    qaggle_angles: np.ndarray
    openfermion_reconstruction_error: float
    qaggle_reconstruction_error: float


@dataclass(frozen=True)
class QROMVectorCollection:
    """The R*B two-body vectors and their section of the ANGLE table."""

    records: tuple[GivensVectorRecord, ...]
    normalized_u: np.ndarray
    qrom_u: np.ndarray
    source_norms: np.ndarray
    qaggle_angles: np.ndarray
    diagonal_signs: np.ndarray
    path_edges: np.ndarray


def _real_float64(array: Any, name: str) -> np.ndarray:
    """Convert a finite real-valued array to float64."""
    value = np.asarray(array)
    if np.iscomplexobj(value):
        if np.max(np.abs(value.imag), initial=0.0) > GIVENS_ATOL:
            raise ValueError(f"{name} must be real")
        value = value.real
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def load_solution(factorization_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load validated parameters and optional diagnostics from an NPZ archive."""
    return read_factorization(factorization_path)


def load_fcidump_reference(fcidump_path: Path) -> dict[str, Any]:
    """Use the dfthc library's FCIDUMP parser."""
    return load_integrals.fci_from_fcidump(str(fcidump_path))["ref"]


def givens_block(theta: float, phi: float) -> np.ndarray:
    """Return the one-particle block documented by OpenFermion."""
    cosine = np.cos(theta)
    sine = np.sin(theta)
    phase = np.exp(1.0j * phi)
    return np.array(
        [[cosine, -phase * sine], [sine, phase * cosine]],
        dtype=complex,
    )


def reconstruct_openfermion_row(
    layers: tuple[GivensLayer, ...],
    left_unitary: np.ndarray,
    diagonal: complex,
    orbitals: int,
) -> np.ndarray:
    r"""Reconstruct q from OpenFermion's relation V q U^dag = D."""
    if left_unitary.shape != (1, 1):
        raise ValueError("this reconstruction expects a single input row")

    result = np.zeros(orbitals, dtype=complex)
    result[0] = left_unitary[0, 0].conjugate() * diagonal
    for layer in reversed(layers):
        for mode_i, mode_j, theta, phi in layer:
            result[[mode_i, mode_j]] = result[[mode_i, mode_j]] @ givens_block(
                theta, phi
            )
    return result


def qaggle_path_vector(
    path_edges: np.ndarray,
    angles: np.ndarray,
    orbitals: int,
) -> npt.NDArray[np.float64]:
    """Apply Qaggle's real coefficient recurrence to e_0."""
    result = np.zeros(orbitals, dtype=np.float64)
    result[0] = 1.0
    for (mode_i, mode_j), theta in zip(path_edges, angles, strict=True):
        value_i = result[mode_i]
        value_j = result[mode_j]
        cosine = np.cos(theta)
        sine = np.sin(theta)
        result[mode_i] = cosine * value_i - sine * value_j
        result[mode_j] = sine * value_i + cosine * value_j
    return result


def decompose_vector(
    vector: np.ndarray,
    rank_index: int,
    basis_index: int,
    source_norm: float,
) -> tuple[GivensVectorRecord, npt.NDArray[np.float64]]:
    """Collect raw OpenFermion data and Qaggle-ready path angles."""
    orbitals = vector.size
    raw_layers, left_unitary, raw_diagonal = of.givens_decomposition(
        vector[None, :],
        always_insert=True,
    )
    layers = tuple(
        tuple((int(i), int(j), float(theta), float(phi)) for i, j, theta, phi in layer)
        for layer in raw_layers
    )
    left_unitary = np.asarray(left_unitary, dtype=complex)
    diagonal = complex(raw_diagonal[0])

    reconstructed = reconstruct_openfermion_row(
        layers,
        left_unitary,
        diagonal,
        orbitals,
    )
    openfermion_error = float(np.max(np.abs(reconstructed - vector)))
    if openfermion_error > GIVENS_ATOL:
        raise AssertionError(
            f"OpenFermion failed to reconstruct U[{rank_index}, {basis_index}]: "
            f"error={openfermion_error:.3e}"
        )

    if not np.allclose(left_unitary, np.ones((1, 1)), atol=GIVENS_ATOL):
        raise ValueError("the one-row decomposition has a nontrivial left unitary")
    if abs(diagonal.imag) > GIVENS_ATOL or not np.isclose(
        abs(diagonal.real),
        1.0,
        atol=GIVENS_ATOL,
    ):
        raise ValueError("a real normalized vector should have diagonal +1 or -1")
    diagonal_sign = 1 if diagonal.real >= 0.0 else -1

    preparation_rotations = [
        rotation for layer in reversed(layers) for rotation in layer
    ]
    path_edges = np.column_stack((np.arange(orbitals - 1), np.arange(1, orbitals)))
    if len(preparation_rotations) != orbitals - 1:
        raise ValueError("expected one path rotation per adjacent mode pair")

    qaggle_angles = np.empty(orbitals - 1, dtype=np.float64)
    for index, (expected_edge, rotation) in enumerate(
        zip(path_edges, preparation_rotations, strict=True)
    ):
        mode_i, mode_j, theta, phi = rotation
        if (mode_i, mode_j) != tuple(expected_edge):
            raise ValueError("OpenFermion did not return the expected path ordering")
        if not np.isclose(np.exp(1.0j * phi), 1.0, atol=GIVENS_ATOL):
            raise ValueError("a real Qaggle vector requires phi = 0")

        # OpenFermion decomposes q; Qaggle prepares q with the inverse action.
        positive_angle = (-theta) % (2.0 * np.pi)
        qaggle_angles[index] = (
            0.0 if positive_angle >= 2.0 * np.pi else positive_angle
        )

    qrom_vector: npt.NDArray[np.float64] = qaggle_path_vector(
        path_edges,
        qaggle_angles,
        orbitals,
    )
    canonical_vector = diagonal_sign * vector
    qaggle_error = float(np.max(np.abs(qrom_vector - canonical_vector)))
    if qaggle_error > GIVENS_ATOL:
        raise AssertionError(
            f"Qaggle angles failed for U[{rank_index}, {basis_index}]: "
            f"error={qaggle_error:.3e}"
        )

    givens_record = GivensVectorRecord(
        rank_index=rank_index,
        basis_index=basis_index,
        source_norm=source_norm,
        vector=vector.copy(),
        layers=layers,
        left_unitary=left_unitary,
        diagonal=diagonal,
        qaggle_angles=qaggle_angles.copy(),
        openfermion_reconstruction_error=openfermion_error,
        qaggle_reconstruction_error=qaggle_error,
    )
    return givens_record, qrom_vector


def collect_qrom_vectors(raw_u: Any) -> QROMVectorCollection:
    """Normalize U[R,B,N] and decompose all R*B distinct vectors."""
    raw_u = _real_float64(raw_u, "U")
    if raw_u.ndim != 3 or raw_u.shape[-1] < 2:
        raise ValueError("U must have shape (R, B, N) with N >= 2")

    # Step 1 (normalize): u[r,b] = U[r,b] / ||U[r,b]||_2.  The archived U is
    # unnormalized; df_thc_sga.Compute_H2_sqrt applies the same normalization.
    source_norms = np.linalg.norm(raw_u, axis=-1)
    if np.any(source_norms <= 0.0) or not np.all(np.isfinite(source_norms)):
        bad = np.argwhere((source_norms <= 0.0) | ~np.isfinite(source_norms))[0]
        raise ValueError(f"U[{bad[0]}, {bad[1]}] cannot be normalized")
    normalized_u = raw_u / source_norms[..., None]

    rank, bases, orbitals = normalized_u.shape
    path_edges = np.column_stack((np.arange(orbitals - 1), np.arange(1, orbitals)))
    qrom_u = np.empty_like(normalized_u)
    qaggle_angles = np.empty((rank, bases, orbitals - 1), dtype=np.float64)
    diagonal_signs = np.empty((rank, bases), dtype=np.int8)
    givens_records: list[GivensVectorRecord] = []

    for rank_index in range(rank):
        for basis_index in range(bases):
            # Steps 2-4 happen per vector inside decompose_vector:
            #   Step 2 (decompose): of.givens_decomposition gives N-1
            #     nearest-neighbour rotations with V u U^dag = D, D = +/-1.
            #   Step 3 (convert): Qaggle angle theta_k = -theta_k^OF mod 2pi;
            #     these N-1 angles are this vector's row of the ANGLE table.
            #   Step 4 (simulate): qaggle_path_vector applies those rotations
            #     to e_0, giving the vector the circuit actually prepares
            #     (= D * u, asserted to GIVENS_ATOL).
            givens_record, qrom_vector = decompose_vector(
                normalized_u[rank_index, basis_index],
                rank_index,
                basis_index,
                float(source_norms[rank_index, basis_index]),
            )
            givens_records.append(givens_record)
            # qrom_u is built only from the angle table (Step 4 output), so
            # comparing its H2_sqrt with the one from U checks the table.
            qrom_u[rank_index, basis_index] = qrom_vector
            qaggle_angles[rank_index, basis_index] = givens_record.qaggle_angles
            # D = -1 means the circuit prepares -u; harmless because only
            # u u^T enters H2_sqrt.
            diagonal_signs[rank_index, basis_index] = (
                1 if givens_record.diagonal.real >= 0.0 else -1
            )

    return QROMVectorCollection(
        records=tuple(givens_records),
        normalized_u=normalized_u,
        qrom_u=qrom_u,
        source_norms=source_norms,
        qaggle_angles=qaggle_angles,
        diagonal_signs=diagonal_signs,
        path_edges=path_edges,
    )


def two_electron_frobenius_error(
    h2_sqrt: Any,
    ref: dict[str, Any],
    hsym_lower: Any,
) -> float:
    """Return the full N^4 Frobenius error using the training-code loss.

    Delegates to ``dfthc.df_thc_sga.Compute_H2_residual_after_symmetry_shift``
    (undoes the particle-number shift, Eq. (40)) and
    ``dfthc.df_thc_sga.Compute_H2_norm`` (restores packed-pair multiplicities).
    """
    h2_residual = df_thc_sga.Compute_H2_residual_after_symmetry_shift(
        {"Hsym_lower": jnp.asarray(_real_float64(hsym_lower, "Hsym_lower"))},
        ref,
        jnp.asarray(h2_sqrt),
    )
    return float(df_thc_sga.Compute_H2_norm(h2_residual, ref["tril_diag"]))


def save_qrom_data(
    output_path: Path,
    system: str,
    collection: QROMVectorCollection,
    weights: np.ndarray,
) -> None:
    """Save the two-body section used after the N one-body ANGLE rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theta_over_pi = np.mod(collection.qaggle_angles / np.pi, 2.0)
    theta_over_pi[theta_over_pi >= 2.0] = 0.0
    np.savez_compressed(
        output_path,
        system=np.asarray(system),
        normalized_u=collection.normalized_u,
        qrom_u=collection.qrom_u,
        u_source_norms=collection.source_norms,
        qaggle_theta=collection.qaggle_angles,
        qaggle_theta_over_pi=theta_over_pi,
        qaggle_path_edges=collection.path_edges,
        openfermion_diagonal_sign=collection.diagonal_signs,
        W=np.asarray(weights, dtype=np.float64),
    )


def _resolve_paths(
    system: str,
    factorization_override: Path | None,
    fcidump_override: Path | None,
) -> tuple[Path, Path]:
    factorization_path = factorization_override or (
        OPTIMIZED_SOLUTIONS_DIRECTORY / OPTIMIZED_SOLUTIONS[system]
    )
    fcidump_path = fcidump_override or (
        INTEGRALS_DIRECTORY / FCIDUMP_FILES[system]
    )
    if not factorization_path.is_file():
        raise FileNotFoundError(factorization_path)
    if not fcidump_path.is_file():
        raise FileNotFoundError(fcidump_path)
    return factorization_path, fcidump_path


def analyze_factorization(
    system: str,
    factorization_path: Path,
    fcidump_path: Path,
    output_path: Path | None,
    skip_loss: bool,
) -> None:
    """Run the complete vector, H2_sqrt, and two-electron checks."""
    params, saved_values = load_solution(factorization_path)
    collection = collect_qrom_vectors(params["U"])
    weights = _real_float64(params["W"], "W")

    # df_thc_sga.Compute_H2_sqrt normalizes U internally, so the raw archive U is
    # the training-code path; the QROM vectors are already unit norm.
    h2_sqrt_from_u = np.asarray(
        df_thc_sga.Compute_H2_sqrt(_real_float64(params["U"], "U"), weights)
    )
    h2_sqrt_from_qrom = np.asarray(
        df_thc_sga.Compute_H2_sqrt(collection.qrom_u, weights)
    )
    h2_sqrt_error = float(
        np.max(np.abs(h2_sqrt_from_qrom - h2_sqrt_from_u), initial=0.0)
    )
    if h2_sqrt_error > H2_SQRT_ATOL:
        raise AssertionError(
            f"QROM vectors do not reconstruct H2_sqrt: {h2_sqrt_error:.3e}"
        )

    rank, bases, orbitals = collection.normalized_u.shape
    copies = weights.shape[-1]
    max_openfermion_error = max(
        givens_record.openfermion_reconstruction_error
        for givens_record in collection.records
    )
    max_qaggle_error = max(
        givens_record.qaggle_reconstruction_error
        for givens_record in collection.records
    )
    negative_signs = int(np.count_nonzero(collection.diagonal_signs < 0))

    print(f"system: {system}")
    print(f"factorization dimensions: R={rank}, B={bases}, C={copies}, N={orbitals}")
    print(f"two-electron QROM vectors: R*B = {rank * bases}")
    print(f"angles per vector: N-1 = {orbitals - 1}")
    print(f"dense QROM angle table shape: {collection.qaggle_angles.shape}")
    print(
        "angle convention: theta_Qaggle = -theta_OpenFermion mod 2pi "
        "(not theta/2)"
    )
    print("angle units: radians; Qaggle fixed-point words encode theta/pi")
    print(f"vectors using the -u global-sign gauge: {negative_signs}")
    print(f"max OpenFermion vector error: {max_openfermion_error:.3e}")
    print(f"max Qaggle recurrence error: {max_qaggle_error:.3e}")
    print(f"H2_sqrt shape: {h2_sqrt_from_qrom.shape}")
    print(f"max H2_sqrt reconstruction error: {h2_sqrt_error:.3e}")

    first_record = collection.records[0]
    preview_count = min(5, first_record.qaggle_angles.size)
    print(
        "first QROM address (r=0, b=0), first angles: "
        f"{first_record.qaggle_angles[:preview_count]}"
    )

    if not skip_loss:
        ref = load_fcidump_reference(fcidump_path)
        if int(ref["orbitals"]) != orbitals:
            raise ValueError("FCIDUMP and factorization have different orbital counts")
        loss_from_u = two_electron_frobenius_error(
            h2_sqrt_from_u, ref, params["Hsym_lower"]
        )
        loss_from_qrom = two_electron_frobenius_error(
            h2_sqrt_from_qrom, ref, params["Hsym_lower"]
        )
        print(f"two-electron loss from normalized U: {loss_from_u:.10f} Ha")
        print(f"two-electron loss from Givens vectors: {loss_from_qrom:.10f} Ha")

        if "error_h2" in saved_values:
            saved_loss = float(np.asarray(saved_values["error_h2"]))
            print(f"saved values['error_h2']: {saved_loss:.10f} Ha")
            print(f"absolute loss difference: {abs(loss_from_u - saved_loss):.3e} Ha")
            if not np.isclose(loss_from_u, saved_loss, atol=5.0e-4, rtol=0.0):
                raise AssertionError("reconstructed loss does not match the archive")

    if output_path is not None:
        save_qrom_data(output_path, system, collection, weights)
        print(f"saved QROM data: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=sorted(OPTIMIZED_SOLUTIONS),
        default="FeMoCo54o54e", # "Fe2S2-20o30e",
        help="archived system to analyze (default: %(default)s)",
    )
    parser.add_argument("--factorization", type=Path, help="override the solution .npz")
    parser.add_argument("--fcidump", type=Path, help="override the FCIDUMP")
    parser.add_argument("--output", type=Path, help="optional QROM-data .npz output")
    parser.add_argument(
        "--skip-loss",
        action="store_true",
        help="collect vectors and reconstruct H2_sqrt without contracting the loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    factorization_path, fcidump_path = _resolve_paths(
        args.system,
        args.factorization,
        args.fcidump,
    )
    analyze_factorization(
        system=args.system,
        factorization_path=factorization_path,
        fcidump_path=fcidump_path,
        output_path=args.output,
        skip_loss=args.skip_loss,
    )


if __name__ == "__main__":
    main()
