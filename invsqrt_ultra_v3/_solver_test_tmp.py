import random
import numpy as np
from typing import Sequence

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
        @njit

        @njit(parallel=True)


def _ints_to_u64x4(vs: Sequence[int]) -> np.ndarray:
    arr = np.zeros((len(vs), 4), dtype=np.uint64)
    for i, v in enumerate(vs):
        arr[i, 0] = np.uint64(v & 0xFFFFFFFFFFFFFFFF)
        arr[i, 1] = np.uint64((v >> 64) & 0xFFFFFFFFFFFFFFFF)
        arr[i, 2] = np.uint64((v >> 128) & 0xFFFFFFFFFFFFFFFF)
        arr[i, 3] = np.uint64((v >> 192) & 0xFFFFFFFFFFFFFFFF)
    return arr



def symplectic_decompose(mat_rows: list[int], M: int) -> tuple[list[tuple[int, int]], int]:
    """Dickson symplectic decomposition of an M x M skew-symmetric GF(2) matrix.

    Args:
        mat_rows: list of M integers (bit j of mat_rows[i] is A[i, j], diagonal 0).
        M: dimension.

    Returns:
        (pairs_uv, diag_fix) where pairs_uv is a list of r = rank(A)//2 pairs (u_mask, v_mask)
        such that:
            XOR_{(u,v)} ((u.w) & (v.w)) ^ (diag_fix . w) == XOR_{i<j} A[i,j] w_i w_j
        and for each pair (u, v), there exists p1 in u\\v and p2 in v\\u (interference-free!).
    """
    A = list(mat_rows)
    pairs_uv: list[tuple[int, int]] = []
    diag_fix = 0
    for i in range(M):
        while A[i]:
            lsb = A[i] & -A[i]
            j = lsb.bit_length() - 1
            u = A[i]  # bit j is 1, bit i is 0
            v = A[j]  # bit i is 1, bit j is 0
            pairs_uv.append((u, v))
            diag_fix ^= (u & v)
            # Subtract u v^T + v u^T (clearing diagonal)
            um = u
            while um:
                lb = um & -um
                a = lb.bit_length() - 1
                A[a] ^= v
                um ^= lb
            vm = v
            while vm:
                lb = vm & -vm
                a = lb.bit_length() - 1
                A[a] ^= u
                vm ^= lb
            for a in range(M):
                A[a] &= ~(1 << a)
    return pairs_uv, diag_fix



def _extract_quad_solution(
    comb: int,
    M: int,
    u_masks: Sequence[int],

def _verify_quad_eval(
    W_vals: Sequence[int],
    pairs_uv: Sequence[tuple[int, int]],
    lin_mask: int,
    const_bit: int,
    n_in: int = 256,

def solve_low_rank_quad(
    W_vals: Sequence[int],
    target_val: int,
    n_in: int = 256,
    rng: random.Random | None = None,
    try_r_below: int = 1,

def _solve_bcast_quad_multi(
    W_vals: Sequence[int],
    prod_vals: Sequence[int],
    rem_targets: Sequence[int],
    n_in: int,
    rng: random.Random,
    trials: int = 18,
    cur_val: int = 0,

