"""Zero-Ancilla GF(2) Quadratic-Form QROM Synthesizer & Measurement-Based (0-Toffoli) Uncomputer.

Synthesizes arbitrary 8-bit lookup tables (N = 256) directly into target registers
using low-symplectic-rank quadratic Boolean forms over the growing wire basis, and
uncomputes registers with ZERO Toffolis via HMR + Clifford CZ/Z/NEG phase fixups
whenever the remaining wires span the target quadratically.
"""

from __future__ import annotations

import random
from typing import Sequence

import kickmix as km
import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit
    def _nb_lsb256(v0, v1, v2, v3):
        if v0 != np.uint64(0):
            x = v0 & (~v0 + np.uint64(1))
            idx = 0
            while x > np.uint64(1):
                x >>= np.uint64(1)
                idx += 1
            return idx
        if v1 != np.uint64(0):
            x = v1 & (~v1 + np.uint64(1))
            idx = 64
            while x > np.uint64(1):
                x >>= np.uint64(1)
                idx += 1
            return idx
        if v2 != np.uint64(0):
            x = v2 & (~v2 + np.uint64(1))
            idx = 128
            while x > np.uint64(1):
                x >>= np.uint64(1)
                idx += 1
            return idx
        if v3 != np.uint64(0):
            x = v3 & (~v3 + np.uint64(1))
            idx = 192
            while x > np.uint64(1):
                x >>= np.uint64(1)
                idx += 1
            return idx
        return -1

    @njit(parallel=True)
    def _nb_search_u_hit_256(W_arr, base_piv_v, base_piv_has, rem_arr, cur_arr, has_cur, r_try, n_trials, seed):
        """16-thread parallel Numba search for random u_masks (r_try,) whose quadratic span hits rem_arr (shifted by cur_arr)."""
        M = W_arr.shape[0]
        d = rem_arr.shape[0]
        n_threads = 16
        per_t = (n_trials + n_threads - 1) // n_threads
        found_flag = np.zeros(1, dtype=np.int32)
        u_wins = np.zeros((n_threads, r_try), dtype=np.int64)
        t_won = np.zeros(n_threads, dtype=np.int8)

        for tid in prange(n_threads):
            # Simple xorshift64* PRNG per thread (thread-safe & blazing fast)
            st = np.uint64(seed + 1) * np.uint64(6364136223846793005) + np.uint64(tid + 1) * np.uint64(1442695040888963407)
            if st == np.uint64(0):
                st = np.uint64(0x9E3779B97F4A7C15)
            u_cur = np.zeros(r_try, dtype=np.int64)
            u_vecs = np.zeros((r_try, 4), dtype=np.uint64)
            piv_v = np.zeros((256, 4), dtype=np.uint64)
            piv_has = np.zeros(256, dtype=np.uint8)
            tp_v = np.zeros((256, 4), dtype=np.uint64)
            tp_has = np.zeros(256, dtype=np.uint8)

            for _tr in range(per_t):
                if found_flag[0] != 0:
                    break
                for i in range(256):
                    piv_has[i] = base_piv_has[i]
                    if piv_has[i]:
                        piv_v[i, 0] = base_piv_v[i, 0]
                        piv_v[i, 1] = base_piv_v[i, 1]
                        piv_v[i, 2] = base_piv_v[i, 2]
                        piv_v[i, 3] = base_piv_v[i, 3]
                mask_m = (np.uint64(1) << np.uint64(M)) - np.uint64(1)
                for k in range(r_try):
                    st ^= st >> np.uint64(12)
                    st ^= st << np.uint64(25)
                    st ^= st >> np.uint64(27)
                    um_u = (st * np.uint64(2685821657736338717)) & mask_m
                    if um_u == np.uint64(0):
                        um_u = np.uint64(1)
                    u_cur[k] = np.int64(um_u)
                    uv0 = np.uint64(0); uv1 = np.uint64(0); uv2 = np.uint64(0); uv3 = np.uint64(0)
                    tmp = um_u
                    a = 0
                    while tmp != np.uint64(0):
                        if tmp & np.uint64(1):
                            uv0 ^= W_arr[a, 0]; uv1 ^= W_arr[a, 1]; uv2 ^= W_arr[a, 2]; uv3 ^= W_arr[a, 3]
                        tmp >>= np.uint64(1)
                        a += 1
                    u_vecs[k, 0] = uv0; u_vecs[k, 1] = uv1; u_vecs[k, 2] = uv2; u_vecs[k, 3] = uv3

                for k in range(r_try):
                    uv0 = u_vecs[k, 0]; uv1 = u_vecs[k, 1]; uv2 = u_vecs[k, 2]; uv3 = u_vecs[k, 3]
                    for j in range(M):
                        v0 = uv0 & W_arr[j, 0]; v1 = uv1 & W_arr[j, 1]; v2 = uv2 & W_arr[j, 2]; v3 = uv3 & W_arr[j, 3]
                        while True:
                            b = _nb_lsb256(v0, v1, v2, v3)
                            if b < 0:
                                break
                            if piv_has[b]:
                                v0 ^= piv_v[b, 0]; v1 ^= piv_v[b, 1]; v2 ^= piv_v[b, 2]; v3 ^= piv_v[b, 3]
                            else:
                                piv_has[b] = 1
                                piv_v[b, 0] = v0; piv_v[b, 1] = v1; piv_v[b, 2] = v2; piv_v[b, 3] = v3
                                break

                # Reduce cur_arr against piv if has_cur
                r0_0 = np.uint64(0); r0_1 = np.uint64(0); r0_2 = np.uint64(0); r0_3 = np.uint64(0)
                r0_nz = False
                if has_cur:
                    r0_0 = cur_arr[0]; r0_1 = cur_arr[1]; r0_2 = cur_arr[2]; r0_3 = cur_arr[3]
                    while True:
                        b = _nb_lsb256(r0_0, r0_1, r0_2, r0_3)
                        if b < 0:
                            break
                        if piv_has[b]:
                            r0_0 ^= piv_v[b, 0]; r0_1 ^= piv_v[b, 1]; r0_2 ^= piv_v[b, 2]; r0_3 ^= piv_v[b, 3]
                        else:
                            r0_nz = True
                            break

                for i in range(256):
                    tp_has[i] = 0
                hit = False
                for ti in range(d):
                    v0 = rem_arr[ti, 0]; v1 = rem_arr[ti, 1]; v2 = rem_arr[ti, 2]; v3 = rem_arr[ti, 3]
                    while True:
                        b = _nb_lsb256(v0, v1, v2, v3)
                        if b < 0:
                            if not r0_nz:
                                hit = True
                            break
                        if piv_has[b]:
                            v0 ^= piv_v[b, 0]; v1 ^= piv_v[b, 1]; v2 ^= piv_v[b, 2]; v3 ^= piv_v[b, 3]
                        elif tp_has[b]:
                            v0 ^= tp_v[b, 0]; v1 ^= tp_v[b, 1]; v2 ^= tp_v[b, 2]; v3 ^= tp_v[b, 3]
                        else:
                            tp_has[b] = 1
                            tp_v[b, 0] = v0; tp_v[b, 1] = v1; tp_v[b, 2] = v2; tp_v[b, 3] = v3
                            break
                    if hit:
                        break
                if (not hit) and r0_nz:
                    v0 = r0_0; v1 = r0_1; v2 = r0_2; v3 = r0_3
                    while True:
                        b = _nb_lsb256(v0, v1, v2, v3)
                        if b < 0:
                            hit = True
                            break
                        if piv_has[b]:
                            v0 ^= piv_v[b, 0]; v1 ^= piv_v[b, 1]; v2 ^= piv_v[b, 2]; v3 ^= piv_v[b, 3]
                        elif tp_has[b]:
                            v0 ^= tp_v[b, 0]; v1 ^= tp_v[b, 1]; v2 ^= tp_v[b, 2]; v3 ^= tp_v[b, 3]
                        else:
                            break
                if hit:
                    found_flag[0] = 1
                    t_won[tid] = 1
                    for k in range(r_try):
                        u_wins[tid, k] = u_cur[k]
                    break

        for tid in range(n_threads):
            if t_won[tid]:
                return True, u_wins[tid]
        return False, u_wins[0]


def _ints_to_u64x4(vs: Sequence[int]) -> np.ndarray:
    arr = np.zeros((len(vs), 4), dtype=np.uint64)
    for i, v in enumerate(vs):
        arr[i, 0] = np.uint64(v & 0xFFFFFFFFFFFFFFFF)
        arr[i, 1] = np.uint64((v >> 64) & 0xFFFFFFFFFFFFFFFF)
        arr[i, 2] = np.uint64((v >> 128) & 0xFFFFFFFFFFFFFFFF)
        arr[i, 3] = np.uint64((v >> 192) & 0xFFFFFFFFFFFFFFFF)
    return arr


def int_tables_to_bit_masks(table: Sequence[int], n_bits: int) -> list[int]:
    """Converts a length-N integer table into `n_bits` N-bit truth-table integer masks."""
    masks = [0] * n_bits
    for idx, val in enumerate(table):
        v = int(val)
        b = 0
        while v:
            lsb = v & -v
            bi = lsb.bit_length() - 1
            if bi < n_bits:
                masks[bi] |= (1 << idx)
            v ^= lsb
    return masks


def addr_bit_masks(n_addr: int = 8) -> list[int]:
    """Returns the `n_addr` truth-table masks for little-endian address bits 0..n_addr-1."""
    N = 1 << n_addr
    out = []
    for b in range(n_addr):
        m = 0
        for x in range(N):
            if (x >> b) & 1:
                m |= (1 << x)
        out.append(m)
    return out


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


def _iter_bits(mask: int) -> list[int]:
    out = []
    m = mask
    while m:
        lsb = m & -m
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out


def apply_and_xor(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    target: km.q,
    u_mask: int,
    v_mask: int,
    u_const: int = 0,
    v_const: int = 0,
) -> None:
    """Applies target ^= ((u.W ^ u_const) & (v.W ^ v_const)) using 1 CCX and 0 ancillas.

    Requires u_mask != 0, v_mask != 0, u_mask != v_mask.
    """
    u_only = u_mask & ~v_mask
    v_only = v_mask & ~u_mask
    if u_only:
        p1 = (u_only & -u_only).bit_length() - 1
        # p1 is NOT in v_mask! Pick any p2 in v_mask (automatically != p1).
        p2 = (v_mask & -v_mask).bit_length() - 1
        # Compute L1 into W[p1] first (may read W[p2], which is still intact!),
        # then compute L2 into W[p2] (never reads W[p1]!).
        u_others = _iter_bits(u_mask ^ (1 << p1))
        v_others = _iter_bits(v_mask ^ (1 << p2))
        for a in u_others:
            builder.cx(W_qubits[a], W_qubits[p1])
        if u_const:
            builder.x(W_qubits[p1])
        for b in v_others:
            builder.cx(W_qubits[b], W_qubits[p2])
        if v_const:
            builder.x(W_qubits[p2])
        builder.ccx(W_qubits[p1], W_qubits[p2], target)
        if v_const:
            builder.x(W_qubits[p2])
        for b in reversed(v_others):
            builder.cx(W_qubits[b], W_qubits[p2])
        if u_const:
            builder.x(W_qubits[p1])
        for a in reversed(u_others):
            builder.cx(W_qubits[a], W_qubits[p1])
    else:
        # u_mask is a strict subset of v_mask -> v_only is guaranteed nonempty!
        assert v_only != 0, 'u_mask and v_mask must be distinct'
        p2 = (v_only & -v_only).bit_length() - 1
        p1 = (u_mask & -u_mask).bit_length() - 1
        v_others = _iter_bits(v_mask ^ (1 << p2))
        u_others = _iter_bits(u_mask ^ (1 << p1))
        for b in v_others:
            builder.cx(W_qubits[b], W_qubits[p2])
        if v_const:
            builder.x(W_qubits[p2])
        for a in u_others:
            builder.cx(W_qubits[a], W_qubits[p1])
        if u_const:
            builder.x(W_qubits[p1])
        builder.ccx(W_qubits[p1], W_qubits[p2], target)
        if u_const:
            builder.x(W_qubits[p1])
        for a in reversed(u_others):
            builder.cx(W_qubits[a], W_qubits[p1])
        if v_const:
            builder.x(W_qubits[p2])
        for b in reversed(v_others):
            builder.cx(W_qubits[b], W_qubits[p2])


def apply_and_xor_bcast(
    builder: km.CircuitBuilder,
    wires: Sequence[km.q],
    targets: Sequence[km.q],
    mask1: int,
    mask2: int,
    c1: int = 0,
    c2: int = 0,
) -> int:
    """Computes `t ^= (L1 & L2)` for EVERY `t` in `targets` using AT MOST 1 CCX total!

    Uses the 0-Toffoli broadcast identity:
      `for tj in targets[1:]: cx(t0, tj)` -> `ccx(p1, p2, t0)` -> `for tj in reversed(targets[1:]): cx(t0, tj)`
    All `targets` must be distinct and disjoint from `wires`.
    """
    assert len(targets) >= 1
    if len(targets) == 1:
        apply_and_xor(builder, wires, targets[0], mask1, mask2, c1, c2)
        return 0 if (mask1 == 0 or mask2 == 0 or mask1 == mask2) else 1

    if mask1 == 0:
        if c1 == 0:
            return 0
        for t in targets:
            if c2:
                builder.x(t)
            for j in _iter_bits(mask2):
                builder.cx(wires[j], t)
        return 0
    if mask2 == 0:
        if c2 == 0:
            return 0
        for t in targets:
            if c1:
                builder.x(t)
            for i in _iter_bits(mask1):
                builder.cx(wires[i], t)
        return 0
    if mask1 == mask2:
        if c1 ^ c2:
            return 0
        for t in targets:
            if c1:
                builder.x(t)
            for i in _iter_bits(mask1):
                builder.cx(wires[i], t)
        return 0

    b1 = _iter_bits(mask1)
    b2 = _iter_bits(mask2)
    diff1 = mask1 & ~mask2
    diff2 = mask2 & ~mask1
    if diff1:
        p1 = _iter_bits(diff1)[0]
        p2 = b2[0]
    else:
        p2 = _iter_bits(diff2)[0]
        p1 = b1[0]

    p2_in_b1 = (mask1 >> p2) & 1

    def _emit_ccx_bcast():
        t0 = targets[0]
        for tj in targets[1:]:
            builder.cx(t0, tj)
        builder.ccx(wires[p1], wires[p2], t0)
        for tj in reversed(targets[1:]):
            builder.cx(t0, tj)

    if not p2_in_b1:
        for j in b2:
            if j != p2:
                builder.cx(wires[j], wires[p2])
        if c2:
            builder.x(wires[p2])
        for i in b1:
            if i != p1:
                builder.cx(wires[i], wires[p1])
        if c1:
            builder.x(wires[p1])
        _emit_ccx_bcast()
        if c1:
            builder.x(wires[p1])
        for i in reversed(b1):
            if i != p1:
                builder.cx(wires[i], wires[p1])
        if c2:
            builder.x(wires[p2])
        for j in reversed(b2):
            if j != p2:
                builder.cx(wires[j], wires[p2])
    else:
        for i in b1:
            if i != p1:
                builder.cx(wires[i], wires[p1])
        if c1:
            builder.x(wires[p1])
        for j in b2:
            if j != p2:
                builder.cx(wires[j], wires[p2])
        if c2:
            builder.x(wires[p2])
        _emit_ccx_bcast()
        if c2:
            builder.x(wires[p2])
        for j in reversed(b2):
            if j != p2:
                builder.cx(wires[j], wires[p2])
        if c1:
            builder.x(wires[p1])
        for i in reversed(b1):
            if i != p1:
                builder.cx(wires[i], wires[p1])
    return 1


def apply_quad_xor(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    target: km.q,
    pairs_uv: Sequence[tuple[int, int]],
    lin_mask: int,
    const_bit: int,
) -> int:
    """Applies target ^= const_bit ^ (lin_mask . W) ^ XOR_{(u,v)} ((u.W) & (v.W)).

    Returns the number of CCX gates emitted (= len(pairs_uv)).
    """
    for u_mask, v_mask in pairs_uv:
        apply_and_xor(builder, W_qubits, target, u_mask, v_mask)
    for a in _iter_bits(lin_mask):
        builder.cx(W_qubits[a], target)
    if const_bit:
        builder.x(target)
    return len(pairs_uv)


def uncompute_quad_hmr(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    target: km.q,
    q_pairs: Sequence[tuple[int, int]],
    lin_mask: int,
    const_bit: int,
) -> None:
    """Uncomputes |target = const ^ lin.W ^ XOR_{(i,j)} W[i]&W[j]> -> |0> using 0 Toffolis!"""
    m = builder.alloc_bits(1)[0]
    builder.hmr(target, m)
    builder.push_condition(m)
    if const_bit:
        builder.neg()
    for i in _iter_bits(lin_mask):
        builder.z(W_qubits[i])
    for i, j in q_pairs:
        builder.cz(W_qubits[i], W_qubits[j])
    builder.pop_condition()


def find_any_quad(
    W_vals: Sequence[int],
    target_val: int,
    n_in: int = 256,
) -> tuple[bool, int, int, list[tuple[int, int]]]:
    """Expresses `target_val` as ANY exact degree-<=2 Boolean function of `W_vals`.

    Used for 0-Toffoli HMR+CZ uncomputation (symplectic rank does not matter).
    Returns (ok, const_bit, lin_mask, q_pairs).
    """
    FULL = (1 << n_in) - 1
    M = len(W_vals)
    pivs: dict[int, int] = {}
    piv_c: dict[int, int] = {}

    # Insert 1 and W_vals first (prefer linear terms!)
    def _insert(v: int, idx: int) -> None:
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
                comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v
                piv_c[lsb] = comb
                return

    _insert(FULL, 0)
    for i in range(M):
        _insert(W_vals[i], 1 + i)

    # Check if already linear (0 CZs!)
    def _try_solve(tv: int):
        t = tv
        tc = 0
        while t:
            lsb = t & -t
            if lsb in pivs:
                t ^= pivs[lsb]
                tc ^= piv_c[lsb]
            else:
                return False, 0
        return True, tc

    ok, tc = _try_solve(target_val)
    if ok:
        const_bit = tc & 1
        lin_mask = (tc >> 1) & ((1 << M) - 1)
        return True, const_bit, lin_mask, []

    # Add quadratic pairs until target is solved or full rank reached!
    pair_list: list[tuple[int, int]] = []
    idx = 1 + M
    for i in range(M):
        wi = W_vals[i]
        for j in range(i + 1, M):
            v = wi & W_vals[j]
            pair_list.append((i, j))
            _insert(v, idx)
            idx += 1
            if len(pivs) == n_in:
                break
        if len(pivs) == n_in:
            break

    ok, tc = _try_solve(target_val)
    if not ok:
        return False, 0, 0, []
    const_bit = tc & 1
    lin_mask = (tc >> 1) & ((1 << M) - 1)
    q_pairs = []
    qbits = tc >> (1 + M)
    while qbits:
        lsb = qbits & -qbits
        pi = lsb.bit_length() - 1
        q_pairs.append(pair_list[pi])
        qbits ^= lsb
    return True, const_bit, lin_mask, q_pairs


def _extract_quad_solution(
    comb: int,
    M: int,
    u_masks: Sequence[int],
) -> tuple[int, int, list[tuple[int, int]]]:
    """Converts a linear-solve bitmask over [1, W_0..W_{M-1}, {U_k & W_j}] into symplectic pairs."""
    const_bit = comb & 1
    lin_mask = (comb >> 1) & ((1 << M) - 1)
    # Build skew-symmetric matrix A (plus diagonal linear correction) from sum_k (u_k.W)*(v_k.W)
    mat = [0] * M
    diag_fix = 0
    qbits = comb >> (1 + M)
    for k, u_m in enumerate(u_masks):
        v_m = (qbits >> (k * M)) & ((1 << M) - 1)
        if not v_m:
            continue
        diag_fix ^= (u_m & v_m)
        um = u_m
        while um:
            lb = um & -um
            a = lb.bit_length() - 1
            mat[a] ^= v_m
            um ^= lb
        vm = v_m
        while vm:
            lb = vm & -vm
            a = lb.bit_length() - 1
            mat[a] ^= u_m
            vm ^= lb
    for a in range(M):
        mat[a] &= ~(1 << a)
    pairs_uv, sd_fix = symplectic_decompose(mat, M)
    lin_final = lin_mask ^ diag_fix ^ sd_fix
    return const_bit, lin_final, pairs_uv


def _verify_quad_eval(
    W_vals: Sequence[int],
    pairs_uv: Sequence[tuple[int, int]],
    lin_mask: int,
    const_bit: int,
    n_in: int = 256,
) -> int:
    FULL = (1 << n_in) - 1
    acc = FULL if const_bit else 0
    for i in _iter_bits(lin_mask):
        acc ^= W_vals[i]
    for u_m, v_m in pairs_uv:
        lv = 0
        for a in _iter_bits(u_m):
            lv ^= W_vals[a]
        rv = 0
        for b in _iter_bits(v_m):
            rv ^= W_vals[b]
        acc ^= (lv & rv)
    return acc


def solve_low_rank_quad(
    W_vals: Sequence[int],
    target_val: int,
    n_in: int = 256,
    rng: random.Random | None = None,
    try_r_below: int = 1,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Finds a low-symplectic-rank quadratic form over `W_vals` equal to `target_val`.

    Uses exact GF(2) linear solving on random U-subspaces of minimal rank r, then applies
    Dickson symplectic decomposition (which can drop rank further if degenerate!).
    Returns (const_bit, lin_mask, pairs_uv).
    """
    if rng is None:
        rng = random.Random(0)
    FULL = (1 << n_in) - 1
    M = len(W_vals)

    # First check r = 0 (linear in 1, W_vals)
    lin_pivs: dict[int, int] = {}
    lin_c: dict[int, int] = {}

    def _ins(pivs, piv_c, v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
                comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v
                piv_c[lsb] = comb
                return

    _ins(lin_pivs, lin_c, FULL, 0)
    for i in range(M):
        _ins(lin_pivs, lin_c, W_vals[i], 1 + i)

    t = target_val
    tc = 0
    while t:
        lsb = t & -t
        if lsb in lin_pivs:
            t ^= lin_pivs[lsb]
            tc ^= lin_c[lsb]
        else:
            break
    if t == 0:
        return tc & 1, (tc >> 1) & ((1 << M) - 1), []

    lin_rk = len(lin_pivs)
    rem = n_in - lin_rk

    # Compute minimum r where 1 + M + r*M - r*(r+1)//2 >= n_in
    r_full = 1
    while r_full < M and lin_rk + r_full * M - r_full * (r_full + 1) // 2 < n_in:
        r_full += 1

    # Try r = r_full (and optionally r_full - 1 if close to threshold)
    best = None
    # Try r_full first with a few random U choices to find a solution (and possibly rank < r_full!)
    for r_try in range(max(1, r_full - try_r_below), r_full + 2):
        n_trials = 6 if r_try == r_full else (3 if r_try < r_full else 10)
        for _ in range(n_trials):
            u_masks = []
            u_vecs = []
            for _k in range(r_try):
                um = rng.randrange(1, 1 << M)
                uv = 0
                for a in _iter_bits(um):
                    uv ^= W_vals[a]
                u_masks.append(um)
                u_vecs.append(uv)
            pivs = dict(lin_pivs)
            piv_c = dict(lin_c)
            idx = 1 + M
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx)
                    idx += 1
            t = target_val
            tc = 0
            while t:
                lsb = t & -t
                if lsb in pivs:
                    t ^= pivs[lsb]
                    tc ^= piv_c[lsb]
                else:
                    break
            if t == 0:
                const_b, lin_m, p_uv = _extract_quad_solution(tc, M, u_masks)
                assert _verify_quad_eval(W_vals, p_uv, lin_m, const_b, n_in) == target_val
                if best is None or len(p_uv) < len(best[2]):
                    best = (const_b, lin_m, p_uv)
                    if len(p_uv) <= r_full - 1:
                        return best
        if best is not None:
            return best

    # Fallback: find_any_quad + symplectic_decompose
    ok, cb, lm, qp = find_any_quad(W_vals, target_val, n_in)
    if not ok:
        raise RuntimeError(f'Target not in quadratic span of {M} wires!')
    mat = [0] * M
    for i, j in qp:
        mat[i] |= (1 << j)
        mat[j] |= (1 << i)
    p_uv, df = symplectic_decompose(mat, M)
    assert _verify_quad_eval(W_vals, p_uv, lm ^ df, cb, n_in) == target_val
    return cb, lm ^ df, p_uv


def _solve_multi_target(
    W_vals: Sequence[int],
    rem_targets: Sequence[int],
    r: int,
    n_in: int,
    rng: random.Random,
    trials: int = 12,
    cur_val: int = 0,
):
    """Finds a nonzero linear combination `goal` of `rem_targets` such that
    `(cur_val ^ goal)` lies in `S(U_1..U_r)` (rank <= r quadratic form over `W_vals`).

    - When `cur_val == 0` (fresh qubit): finds any dependency among `rem_targets` mod `S(U)`.
    - When `cur_val != 0` (in-place overwrite): inserts all `rem_targets` mod `S(U)` FIRST,
      then reduces `cur_val` against `S(U) + span(rem_targets)` so `cur_val` is ALWAYS used!

    Returns (target_comb_mask, const_bit, lin_mask, pairs_uv) or None.
    """
    FULL = (1 << n_in) - 1
    M = len(W_vals)
    lin_pivs: dict[int, int] = {}
    lin_c: dict[int, int] = {}

    def _ins(pivs, piv_c, v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
                comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v
                piv_c[lsb] = comb
                return

    _ins(lin_pivs, lin_c, FULL, 0)
    for i in range(M):
        _ins(lin_pivs, lin_c, W_vals[i], 1 + i)

    best = None
    for _ in range(trials):
        u_masks = []
        u_vecs = []
        for _k in range(r):
            um = rng.randrange(1, 1 << M)
            uv = 0
            for a in _iter_bits(um):
                uv ^= W_vals[a]
            u_masks.append(um)
            u_vecs.append(uv)
        pivs = dict(lin_pivs)
        piv_c = dict(lin_c)
        idx = 1 + M
        for uv in u_vecs:
            for w in W_vals:
                _ins(pivs, piv_c, uv & w, idx)
                idx += 1

        t_pivs: dict[int, int] = {}
        t_cs: dict[int, int] = {}
        t_ct: dict[int, int] = {}
        for ti, tv in enumerate(rem_targets):
            v = tv
            cs = 0
            ct = 1 << ti
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                    cs ^= piv_c[lsb]
                elif lsb in t_pivs:
                    v ^= t_pivs[lsb]
                    cs ^= t_cs[lsb]
                    ct ^= t_ct[lsb]
                else:
                    t_pivs[lsb] = v
                    t_cs[lsb] = cs
                    t_ct[lsb] = ct
                    break
            else:
                if cur_val == 0:
                    cb, lm, p_uv = _extract_quad_solution(cs, M, u_masks)
                    if best is None or len(p_uv) < len(best[3]):
                        best = (ct, cb, lm, p_uv)
                        if len(p_uv) <= r - 1:
                            return best

        if cur_val != 0:
            v = cur_val
            cs = 0
            ct = 0
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                    cs ^= piv_c[lsb]
                elif lsb in t_pivs:
                    v ^= t_pivs[lsb]
                    cs ^= t_cs[lsb]
                    ct ^= t_ct[lsb]
                else:
                    break
            else:
                if ct != 0:
                    cb, lm, p_uv = _extract_quad_solution(cs, M, u_masks)
                    if best is None or len(p_uv) < len(best[3]):
                        best = (ct, cb, lm, p_uv)
                        if len(p_uv) <= r - 1:
                            return best

        if best is not None:
            return best
    return None


def _gf2_linear_transform_inplace(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    addr_vals: Sequence[int],
    out_qubits: Sequence[km.q],
    cur_vals: list[int],
    target_vals: Sequence[int],
    n_in: int = 256,
) -> None:
    """Applies CX, X, SWAP on `out_qubits` (reading `addr_qubits` read-only) so
    `out_qubits[j]` ends up holding `target_vals[j]`!

    Requires that `target_vals[j] in span(1, addr_vals, cur_vals)` and `target_vals`
    spans the same quotient space modulo `span(1, addr_vals)` as `cur_vals` (invertible!).
    Wait: even if `target_vals` has linear dependencies among itself/addr, we handle it
    via exact linear algebra if `cur_vals` is an invertible transformation of `target_vals`
    modulo `span(1, addr_vals)`.
    """
    FULL = (1 << n_in) - 1
    m = len(out_qubits)
    n_a = len(addr_qubits)
    # Express each target_vals[j] as const_j ^ (A_j . addr) ^ (B_j . cur_vals)
    pivs: dict[int, int] = {}
    piv_c: dict[int, int] = {}

    def _ins(v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
                comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v
                piv_c[lsb] = comb
                return

    _ins(FULL, 0)
    for i in range(n_a):
        _ins(addr_vals[i], 1 + i)
    for i in range(m):
        _ins(cur_vals[i], 1 + n_a + i)

    consts = [0] * m
    a_rows = [0] * m
    b_rows = [0] * m  # m x m invertible binary matrix B: target = B * cur ^ A * addr ^ const
    for j in range(m):
        t = target_vals[j]
        tc = 0
        while t:
            lsb = t & -t
            if lsb not in pivs:
                raise RuntimeError(f'Target bit {j} not in span(1, addr, cur_vals)!')
            t ^= pivs[lsb]
            tc ^= piv_c[lsb]
        consts[j] = tc & 1
        a_rows[j] = (tc >> 1) & ((1 << n_a) - 1)
        b_rows[j] = (tc >> (1 + n_a)) & ((1 << m) - 1)

    # Synthesize in-place invertible matrix B on `out_qubits` using Gauss-Jordan!
    # We want to apply a sequence of elementary row operations E_k ... E_1 to `cur`
    # such that `cur` becomes `B * cur`.
    # Equivalently, reduce `B` to `I` using row operations `R_p ... R_1 B = I`,
    # then apply the INVERSE operations `R_1^{-1} ... R_p^{-1}` to `out_qubits`!
    B = list(b_rows)
    ops: list[tuple[str, int, int]] = []
    for col in range(m):
        piv = None
        for r in range(col, m):
            if (B[r] >> col) & 1:
                piv = r
                break
        if piv is None:
            raise RuntimeError('Matrix B is singular in linear cleanup!')
        if piv != col:
            B[col], B[piv] = B[piv], B[col]
            ops.append(('SWAP', col, piv))
        for r in range(m):
            if r != col and ((B[r] >> col) & 1):
                B[r] ^= B[col]
                ops.append(('CX', col, r))  # row r ^= row col

    # Apply ops in REVERSE order (since each CX(c, r) and SWAP(a, b) is self-inverse!)
    for kind, a, b in reversed(ops):
        if kind == 'SWAP':
            builder.swap(out_qubits[a], out_qubits[b])
            cur_vals[a], cur_vals[b] = cur_vals[b], cur_vals[a]
        else:
            builder.cx(out_qubits[a], out_qubits[b])
            cur_vals[b] ^= cur_vals[a]

    # Now add A * addr and const directly into out_qubits!
    for j in range(m):
        for i in _iter_bits(a_rows[j]):
            builder.cx(addr_qubits[i], out_qubits[j])
            cur_vals[j] ^= addr_vals[i]
        if consts[j]:
            builder.x(out_qubits[j])
            cur_vals[j] ^= FULL
        assert cur_vals[j] == target_vals[j], f'Linear cleanup mismatch at bit {j}'


def synthesize_qrom1(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    out_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    n_boot: int | None = None,
    seed: int = 0,
) -> int:
    """Zero-ancilla (beyond optional `temp_qubits`) GF(2) quadratic-form QROM synthesizer.

    Computes `out_qubits[j] = target_vals[j](addr)` for j = 0..m-1 (`len(addr_qubits) == 8`),
    and uncomputes `temp_qubits` back to |0> using 0-Toffoli HMR+CZ!

    Strategy:
      1. Initialize `n_boot` qubits of `out_qubits` + all `temp_qubits` with 1-CCX nonlinear
         basis products (maximizing linear independence!).
      2. Compute remaining `(m - n_boot)` fresh `out_qubits` directly using low-rank quadratic
         forms (`_solve_multi_target` to hit ANY vector in the remaining target span!).
      3. Overwrite the `n_boot` bootstrap qubits of `out_qubits` in-place to complete the
         target subspace span.
      4. Apply 0-Toffoli CX/SWAP/X linear cleanup to place exact `target_vals` on `out_qubits`.
      5. Uncompute `temp_qubits` via 0-Toffoli HMR+CZ.

    Returns total CCX count emitted.
    """
    n_addr = len(addr_qubits)
    n_in = 1 << n_addr
    FULL = (1 << n_in) - 1
    m = len(out_qubits)
    n_tmp = len(temp_qubits)
    rng = random.Random(seed)
    a_vals = addr_bit_masks(n_addr)

    # Check linear independence of target_vals modulo span(1, addr)
    base_pivs: dict[int, int] = {}

    def _add_piv(pivs, v):
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
            else:
                pivs[lsb] = v
                return True
        return False

    _add_piv(base_pivs, FULL)
    for v in a_vals:
        _add_piv(base_pivs, v)

    # Filter target_vals into linearly independent core (modulo 1, addr) + dependent targets
    # (Dependent targets take 0 CCXs at the end via CX!)
    indep_indices: list[int] = []
    dep_indices: list[int] = []
    t_pivs = dict(base_pivs)
    for j, tv in enumerate(target_vals):
        if _add_piv(t_pivs, tv):
            indep_indices.append(j)
        else:
            dep_indices.append(j)

    m_indep = len(indep_indices)
    # Work on core_qubits = [out_qubits[j] for j in indep_indices], then fill dep_indices with CX!
    core_qubits = [out_qubits[j] for j in indep_indices]
    core_targets = [target_vals[j] for j in indep_indices]

    # Choose default n_boot: enough wires so M_boot = n_addr + n_boot + n_tmp >= 28..32
    if n_boot is None:
        # Optimal sweet spot: start direct writes once M ~ 32..36 (or m_indep if smaller)
        n_boot = min(m_indep, max(0, 32 - n_addr - n_tmp))
    n_boot = min(n_boot, m_indep)

    ccx_total = 0
    W_q: list[km.q] = list(addr_qubits)
    W_v: list[int] = list(a_vals)
    w_pivs = dict(base_pivs)

    # Step 1: Bootstrap `temp_qubits` + first `n_boot` of `core_qubits` with 1 CCX each!
    boot_slots: list[tuple[km.q, bool]] = [(q, True) for q in temp_qubits] + [
        (core_qubits[i], False) for i in range(n_boot)
    ]
    boot_wire_indices: list[int] = []  # indices in W_q of core_qubits that need overwriting
    temp_wire_indices: list[int] = []  # indices in W_q of temp_qubits

    for q, is_tmp in boot_slots:
        M_cur = len(W_q)
        # Pick random affine product that is linearly independent of current W_v
        for _ in range(100):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc = rng.randrange(2)
            vc = rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= W_v[a]
            rv = FULL if vc else 0
            for b in _iter_bits(vm):
                rv ^= W_v[b]
            prod = lv & rv
            if _add_piv(dict(w_pivs), prod):
                _add_piv(w_pivs, prod)
                apply_and_xor(builder, W_q, q, um, vm, uc, vc)
                ccx_total += 1
                idx_w = len(W_q)
                W_q.append(q)
                W_v.append(prod)
                if is_tmp:
                    temp_wire_indices.append(idx_w)
                else:
                    boot_wire_indices.append(idx_w)
                break
        else:
            raise RuntimeError('Failed to find independent 1-CCX bootstrap wire')

    # Maintain remaining target subspace (list of target vectors not yet in span(1, addr, done_core))
    done_pivs = dict(base_pivs)
    rem_targets = list(core_targets)

    def _prune_rem(new_vec: int) -> None:
        nonlocal rem_targets
        _add_piv(done_pivs, new_vec)
        # Reduce rem_targets modulo done_pivs and keep a linearly independent basis of remaining targets
        new_rem = []
        tmp_p = dict(done_pivs)
        for tv in rem_targets:
            if _add_piv(tmp_p, tv):
                new_rem.append(tv)
        rem_targets = new_rem

    def _pick_and_solve(avail_v: list[int], cur_val: int) -> tuple[int, int, int, list[tuple[int, int]]]:
        """Picks a target vector `goal` in span(rem_targets) (so we write `delta = cur_val ^ goal`)
        using minimum CCXs! Returns (goal_vec, const_bit, lin_mask, pairs_uv)."""
        M_av = len(avail_v)
        d = len(rem_targets)
        # Compute theoretical r_full for single target and multi-target!
        r_single = 1
        while r_single < M_av and (1 + M_av) + r_single * M_av - r_single * (r_single + 1) // 2 < n_in:
            r_single += 1
        r_multi = 1
        while r_multi < M_av and (1 + M_av) + r_multi * M_av - r_multi * (r_multi + 1) // 2 + d < n_in + 2:
            r_multi += 1

        # Multi-target subspace intersection works for BOTH cur_val == 0 and cur_val != 0!
        if r_multi < r_single and d >= 2:
            for r_try in range(r_multi, r_single):
                hit = _solve_multi_target(avail_v, rem_targets, r_try, n_in, rng, trials=12, cur_val=cur_val)
                if hit is not None:
                    ct, cb, lm, p_uv = hit
                    goal = 0
                    for ti in _iter_bits(ct):
                        goal ^= rem_targets[ti]
                    assert _verify_quad_eval(avail_v, p_uv, lm, cb, n_in) == (cur_val ^ goal)
                    return goal, cb, lm, p_uv

        # Standard single-target solve for rem_targets[0]
        goal = rem_targets[0]
        delta = cur_val ^ goal
        cb, lm, p_uv = solve_low_rank_quad(avail_v, delta, n_in=n_in, rng=rng, try_r_below=1)
        return goal, cb, lm, p_uv

    # Step 2: Compute remaining fresh core_qubits[n_boot..m_indep-1] directly!
    for i in range(n_boot, m_indep):
        q = core_qubits[i]
        goal, cb, lm, p_uv = _pick_and_solve(W_v, 0)
        ccx_total += apply_quad_xor(builder, W_q, q, p_uv, lm, cb)
        W_q.append(q)
        W_v.append(goal)
        _prune_rem(goal)

    # Step 3: Overwrite the `n_boot` bootstrap wires of `core_qubits` in-place!
    for w_idx in boot_wire_indices:
        q = W_q[w_idx]
        cur_v = W_v[w_idx]
        avail_q = W_q[:w_idx] + W_q[w_idx + 1:]
        avail_v = W_v[:w_idx] + W_v[w_idx + 1:]
        goal, cb, lm, p_uv = _pick_and_solve(avail_v, cur_v)
        ccx_total += apply_quad_xor(builder, avail_q, q, p_uv, lm, cb)
        W_v[w_idx] = goal
        _prune_rem(goal)

    assert len(rem_targets) == 0, f'{len(rem_targets)} targets still unreached!'

    # Step 4: Uncompute `temp_qubits` via 0-Toffoli HMR+CZ BEFORE linear cleanup
    # (or after — doing it while all core_qubits + addr_qubits are alive guarantees full rank!)
    # Remove temp wires one by one from W_q/W_v
    for w_idx in reversed(temp_wire_indices):
        q = W_q.pop(w_idx)
        tv = W_v.pop(w_idx)
        ok, cb, lm, qp = find_any_quad(W_v, tv, n_in=n_in)
        if not ok:
            raise RuntimeError('Failed to 0-Toffoli HMR uncompute temp qubit!')
        uncompute_quad_hmr(builder, W_q, q, qp, lm, cb)

    # Now W_q == addr_qubits + core_qubits (in exact order!), W_v == a_vals + cur_core_vals
    cur_core_vals = W_v[n_addr:]
    assert len(cur_core_vals) == m_indep

    # Step 5: Exact 0-Toffoli CX/SWAP/X linear cleanup on `core_qubits` -> `core_targets`!
    _gf2_linear_transform_inplace(builder, addr_qubits, a_vals, core_qubits, cur_core_vals, core_targets, n_in=n_in)

    # Step 6: Fill any linearly dependent outputs in `dep_indices` via pure CX/X (0 CCXs)!
    if dep_indices:
        pivs: dict[int, int] = {}
        piv_c: dict[int, int] = {}

        def _ins(v, idx):
            comb = 1 << idx
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                    comb ^= piv_c[lsb]
                else:
                    pivs[lsb] = v
                    piv_c[lsb] = comb
                    return

        _ins(FULL, 0)
        for i in range(n_addr):
            _ins(a_vals[i], 1 + i)
        for i in range(m_indep):
            _ins(core_targets[i], 1 + n_addr + i)
        src_q = list(addr_qubits) + list(core_qubits)
        for j in dep_indices:
            t = target_vals[j]
            tc = 0
            while t:
                lsb = t & -t
                t ^= pivs[lsb]
                tc ^= piv_c[lsb]
            if tc & 1:
                builder.x(out_qubits[j])
            for k in _iter_bits(tc >> 1):
                builder.cx(src_q[k], out_qubits[j])

    return ccx_total


def _solve_bcast_quad_multi(
    W_vals: Sequence[int],
    prod_vals: Sequence[int],
    rem_targets: Sequence[int],
    n_in: int,
    rng: random.Random,
    trials: int = 18,
    cur_val: int = 0,
) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    """Finds a nonzero target combo `ct` in `rem_targets` such that
      `cur_val ^ (sum_{ti in ct} rem_targets[ti])`
    is expressible as:
      `const ^ lin(W_vals) ^ bcast(prod_vals) ^ sum_{k=1}^r (U_k & V_k)`
    with minimum `r` (new CCXs), reusing any subset of `prod_vals` for 0 CCXs via broadcast!

    Returns `(target_comb_mask, const_bit, lin_mask, bcast_mask, pairs_uv)`.
    """
    FULL = (1 << n_in) - 1
    M = len(W_vals)
    P = len(prod_vals)
    d = len(rem_targets)
    assert d >= 1

    base_pivs: dict[int, int] = {}
    base_c: dict[int, int] = {}

    def _ins(pivs, piv_c, v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]
                comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v
                piv_c[lsb] = comb
                return

    def _check_hit(pivs, piv_c):
        # Returns (ct, cs) for nonzero ct with cur_val ^ sum(rem[ct]) in span(pivs), or None.
        # First reduce cur_val against pivs -> (r0, cs0).
        r0 = cur_val
        cs0 = 0
        while r0:
            lsb = r0 & -r0
            if lsb in pivs:
                r0 ^= pivs[lsb]
                cs0 ^= piv_c[lsb]
            else:
                break
        tp: dict[int, int] = {}
        tcs: dict[int, int] = {}
        tct: dict[int, int] = {}
        for ti, tv in enumerate(rem_targets):
            v = tv
            cs = 0
            ct = 1 << ti
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                    cs ^= piv_c[lsb]
                elif lsb in tp:
                    v ^= tp[lsb]
                    cs ^= tcs[lsb]
                    ct ^= tct[lsb]
                else:
                    tp[lsb] = v
                    tcs[lsb] = cs
                    tct[lsb] = ct
                    break
            else:
                if r0 == 0:
                    return ct, cs0 ^ cs
        if r0 != 0:
            v = r0
            cs = cs0
            ct = 0
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                    cs ^= piv_c[lsb]
                elif lsb in tp:
                    v ^= tp[lsb]
                    cs ^= tcs[lsb]
                    ct ^= tct[lsb]
                else:
                    return None
            if ct != 0:
                return ct, cs
        return None

    _ins(base_pivs, base_c, FULL, 0)
    for i in range(M):
        _ins(base_pivs, base_c, W_vals[i], 1 + i)
    for p in range(P):
        _ins(base_pivs, base_c, prod_vals[p], 1 + M + p)

    # Check r = 0 first (pure linear + broadcast, 0 new CCXs!)
    h0 = _check_hit(base_pivs, base_c)
    if h0 is not None:
        ct, cs = h0
        cb = cs & 1
        lm = (cs >> 1) & ((1 << M) - 1)
        bm = (cs >> (1 + M)) & ((1 << P) - 1)
        return ct, cb, lm, bm, []

    lin_rk = len(base_pivs)
    use_nb = _HAS_NUMBA and (n_in == 256) and (M <= 62)
    max_def = 13 if use_nb else 5
    r_m = 1
    while r_m < M and lin_rk + r_m * M - r_m * (r_m + 1) // 2 + d < n_in - max_def:
        r_m += 1
    r_s = 1
    while r_s < M and lin_rk + r_s * M - r_s * (r_s + 1) // 2 < n_in:
        r_s += 1

    q_off = 1 + M + P
    best = None

    if use_nb:
        W_arr = _ints_to_u64x4(W_vals)
        rem_arr = _ints_to_u64x4(rem_targets)
        cur_arr = _ints_to_u64x4([cur_val])[0]
        bp_v = np.zeros((256, 4), dtype=np.uint64)
        bp_h = np.zeros(256, dtype=np.uint8)
        for lsb, pv in base_pivs.items():
            bi = lsb.bit_length() - 1
            bp_h[bi] = 1
            bp_v[bi, 0] = np.uint64(pv & 0xFFFFFFFFFFFFFFFF)
            bp_v[bi, 1] = np.uint64((pv >> 64) & 0xFFFFFFFFFFFFFFFF)
            bp_v[bi, 2] = np.uint64((pv >> 128) & 0xFFFFFFFFFFFFFFFF)
            bp_v[bi, 3] = np.uint64((pv >> 192) & 0xFFFFFFFFFFFFFFFF)

    for r_try in range(r_m, r_s + 2):
        gap = n_in - (lin_rk + r_try * M - r_try * (r_try + 1) // 2 + d)
        if use_nb:
            n_tr = min(20480, max(64, 3 << max(0, gap))) if gap > 0 else max(trials, 32)
            ok, u_nb = _nb_search_u_hit_256(
                W_arr, bp_v, bp_h, rem_arr, cur_arr, cur_val != 0, r_try, n_tr, rng.randrange(1 << 30)
            )
            if not ok:
                continue
            u_masks = [int(u_nb[k]) for k in range(r_try)]
            u_vecs = []
            for um in u_masks:
                uv = 0
                for a in _iter_bits(um):
                    uv ^= W_vals[a]
                u_vecs.append(uv)
            pivs = dict(base_pivs)
            piv_c = dict(base_c)
            idx = q_off
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx)
                    idx += 1
            hit = _check_hit(pivs, piv_c)
            assert hit is not None, 'Numba hit verification failed!'
            ct, cs = hit
            cb0 = cs & 1
            lm0 = (cs >> 1) & ((1 << M) - 1)
            bm0 = (cs >> (1 + M)) & ((1 << P) - 1)
            synth_c = cb0 | (lm0 << 1) | ((cs >> q_off) << (1 + M))
            cb, lm, p_uv = _extract_quad_solution(synth_c, M, u_masks)
            return ct, cb, lm, bm0, p_uv

        n_tr = max(trials, 56) if gap > 0 else (trials if r_try <= r_s else 12)
        for _ in range(n_tr):
            u_masks = []
            u_vecs = []
            for _k in range(r_try):
                um = rng.randrange(1, 1 << M)
                uv = 0
                for a in _iter_bits(um):
                    uv ^= W_vals[a]
                u_masks.append(um)
                u_vecs.append(uv)
            pivs = dict(base_pivs)
            piv_c = dict(base_c)
            idx = q_off
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx)
                    idx += 1
            hit = _check_hit(pivs, piv_c)
            if hit is not None:
                ct, cs = hit
                cb0 = cs & 1
                lm0 = (cs >> 1) & ((1 << M) - 1)
                bm0 = (cs >> (1 + M)) & ((1 << P) - 1)
                synth_c = cb0 | (lm0 << 1) | ((cs >> q_off) << (1 + M))
                cb, lm, p_uv = _extract_quad_solution(synth_c, M, u_masks)
                if best is None or len(p_uv) < len(best[4]):
                    best = (ct, cb, lm, bm0, p_uv)
                    if len(p_uv) <= r_try - 1:
                        return best
            if best is not None:
                return best

    # Fallback: solve cur_val ^ rem_targets[0] via solve_low_rank_quad
    cb, lm, p_uv = solve_low_rank_quad(W_vals, cur_val ^ rem_targets[0], n_in=n_in, rng=rng, try_r_below=1)
    return 1, cb, lm, 0, p_uv


def compute_quad_bits_bcast(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    W_vals: Sequence[int],
    new_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    n_in: int = 256,
    seed: int = 0,
    use_init_and_temp: bool = False,
    keep_temp_out: dict | None = None,
) -> int:
    """Computes `new_qubits[j] = target_vals[j]` using BOTH multi-target intersection AND
    shared-product CCX broadcasting (all CCX products emitted for earlier targets in the batch
    are reused for 0 CCXs by later targets via `CX t0 tj; CCX; CX t0 tj`)!"""
    FULL = (1 << n_in) - 1
    rng = random.Random(seed)
    cur_q = list(W_qubits)
    cur_v = list(W_vals)
    ccx_total = 0

    pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(pivs, FULL)
    for v in cur_v:
        _add_p(pivs, v)

    # Step 1: Boost basis with 1-CCX random products into temp_qubits (already in cur_v!)
    tmp_vals: list[int] = []
    tmp_init_recs: list[tuple[km.q, int, int, int, int, int]] = []
    for tq in temp_qubits:
        M_cur = len(cur_q)
        for _ in range(100):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc, vc = rng.randrange(2), rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = FULL if vc else 0
            for b in _iter_bits(vm):
                rv ^= cur_v[b]
            prod = lv & rv
            if _add_p(dict(pivs), prod):
                _add_p(pivs, prod)
                if use_init_and_temp:
                    _apply_init_and_linear(builder, cur_q, tq, um, vm, uc, vc, uncompute=False)
                    tmp_init_recs.append((tq, M_cur, um, vm, uc, vc))
                else:
                    apply_and_xor(builder, cur_q, tq, um, vm, uc, vc)
                ccx_total += 1
                cur_q.append(tq)
                cur_v.append(prod)
                tmp_vals.append(prod)
                break

    # Partition target_vals into linearly independent core (modulo 1, cur_v) + dependent (0 CCX!)
    base_pivs = dict(pivs)
    indep_idx: list[int] = []
    dep_idx: list[int] = []
    tp_chk = dict(base_pivs)
    for j, tv in enumerate(target_vals):
        if _add_p(tp_chk, tv):
            indep_idx.append(j)
        else:
            dep_idx.append(j)

    core_qs = [new_qubits[j] for j in indep_idx]
    core_ts = [target_vals[j] for j in indep_idx]

    # Step 2: Plan broadcast CCXs + linear ops across all `core_qs`!
    ops_plan: list[dict] = []
    prod_vals: list[int] = []
    prod_op_idx: list[int] = []

    done_p = dict(base_pivs)
    rem = list(core_ts)
    built_v: list[int] = []

    for q in core_qs:
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(cur_v, prod_vals, rem, n_in=n_in, rng=rng)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]

        # Attach retroactive broadcast targets!
        bcast_val = 0
        for pi in _iter_bits(bm):
            ops_plan[prod_op_idx[pi]]['targets'].append(q)
            bcast_val ^= prod_vals[pi]

        # Verify math before recording!
        q_val = _verify_quad_eval(cur_v, p_uv, lm, cb, n_in)
        assert (q_val ^ bcast_val) == goal, 'Broadcast quad verification failed!'

        snap_q = list(cur_q)
        for um, vm in p_uv:
            lv = 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = 0
            for b in _iter_bits(vm):
                rv ^= cur_v[b]
            pv = lv & rv
            op_idx = len(ops_plan)
            ops_plan.append({
                'type': 'ccx', 'wires': snap_q, 'targets': [q],
                'm1': um, 'm2': vm,
            })
            prod_vals.append(pv)
            prod_op_idx.append(op_idx)
        ops_plan.append({
            'type': 'lin', 'wires': snap_q, 'target': q,
            'lin_mask': lm, 'const_bit': cb,
        })

        cur_q.append(q)
        cur_v.append(goal)
        built_v.append(goal)
        _add_p(done_p, goal)
        new_rem = []
        tp2 = dict(done_p)
        for tv in rem:
            if _add_p(tp2, tv):
                new_rem.append(tv)
        rem = new_rem

    # Step 3: Emit planned ops into builder!
    for op in ops_plan:
        if op['type'] == 'ccx':
            ccx_total += apply_and_xor_bcast(builder, op['wires'], op['targets'], op['m1'], op['m2'])
        else:
            t = op['target']
            for i in _iter_bits(op['lin_mask']):
                builder.cx(op['wires'][i], t)
            if op['const_bit']:
                builder.x(t)

    # Step 4: Exact 0-Tof linear cleanup on `core_qs` -> `core_ts`!
    n_src = len(W_qubits) + len(temp_qubits)
    if core_qs:
        _gf2_linear_transform_inplace(
            builder, cur_q[:n_src], cur_v[:n_src], core_qs, built_v, core_ts, n_in=n_in
        )
        cur_v[n_src:] = list(core_ts)

    # Step 5: Fill any dependent outputs in `dep_idx` via pure CX/X (0 CCX)!
    if dep_idx:
        dp: dict[int, int] = {}
        dc: dict[int, int] = {}

        def _ins_d(v, idx):
            comb = 1 << idx
            while v:
                lsb = v & -v
                if lsb in dp:
                    v ^= dp[lsb]
                    comb ^= dc[lsb]
                else:
                    dp[lsb] = v
                    dc[lsb] = comb
                    return

        _ins_d(FULL, 0)
        for i, v in enumerate(cur_v):
            _ins_d(v, 1 + i)
        for j in dep_idx:
            t = target_vals[j]
            tc = 0
            while t:
                lsb = t & -t
                t ^= dp[lsb]
                tc ^= dc[lsb]
            if tc & 1:
                builder.x(new_qubits[j])
            for k in _iter_bits(tc >> 1):
                builder.cx(cur_q[k], new_qubits[j])

    # Step 6: Free `temp_qubits` via 0-Toffoli `del_and` or `HMR+CZ` (unless caller keeps them alive)!
    if temp_qubits:
        if keep_temp_out is not None:
            keep_temp_out['recs'] = tmp_init_recs
            keep_temp_out['cur_q'] = list(cur_q[:n_src])
            keep_temp_out['cur_v'] = list(cur_v[:n_src])
            keep_temp_out['tmp_vals'] = list(tmp_vals)
        elif use_init_and_temp:
            for tq, M_prev, um, vm, uc, vc in reversed(tmp_init_recs):
                _apply_init_and_linear(builder, cur_q[:M_prev], tq, um, vm, uc, vc, uncompute=True)
        else:
            keep_q = list(W_qubits) + list(new_qubits)
            keep_v = list(W_vals) + list(target_vals)
            uncompute_bits_hmr(builder, keep_q, keep_v, temp_qubits, tmp_vals, n_in=n_in)

    return ccx_total


def synthesize_qrom1_overwrite(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    out_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    k_fresh: int = 6,
    k_hold: int = 0,
    seed: int = 0,
) -> int:
    """v34-style Boot + In-Place Overwrite + Safe Broadcast QROM1!

    Phase 1: Boot `temp_qubits` + `core_qs[:n_boot]` (all except `k_fresh` qubits!) with 1 CCX each,
             immediately jumping wire count M to `n_addr + m_indep + n_tmp - k_fresh` (~39-40 wires!).
    Phase 2: Compute `k_fresh` fresh outputs with full BCAST (M grows to ~45-46 wires!).
    Phase 3: In-place overwrite all `n_boot` bootstrap qubits at M_src ~ 44 wires, reusing every
             earlier CCX product emitted after `last_read_op[q]` for 0 CCXs via broadcast!
    Phase 4: 0-Tof linear cleanup, 0-Tof dependent outputs, 0-Tof HMR+CZ free of `temp_qubits`.
    """
    n_addr = len(addr_qubits)
    n_in = 1 << n_addr
    FULL = (1 << n_in) - 1
    rng = random.Random(seed)
    a_vals = addr_bit_masks(n_addr)

    base_pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(base_pivs, FULL)
    for v in a_vals:
        _add_p(base_pivs, v)

    indep_idx: list[int] = []
    dep_idx: list[int] = []
    t_pivs = dict(base_pivs)
    for j, tv in enumerate(target_vals):
        if _add_p(t_pivs, tv):
            indep_idx.append(j)
        else:
            dep_idx.append(j)

    core_qs = [out_qubits[j] for j in indep_idx]
    core_ts = [target_vals[j] for j in indep_idx]
    m_indep = len(core_qs)
    n_tmp = len(temp_qubits)
    k_fresh = max(1, min(k_fresh, m_indep))
    n_boot = m_indep - k_fresh

    # Phase 1: 1-CCX bootstrap all `temp_qubits` + `core_qs[:n_boot]`!
    ccx_total = 0
    cur_q: list[km.q] = list(addr_qubits)
    cur_v: list[int] = list(a_vals)
    w_pivs = dict(base_pivs)
    tmp_vals: list[int] = []

    for q, is_tmp in [(tq, True) for tq in temp_qubits] + [(bq, False) for bq in core_qs[:n_boot]]:
        M_cur = len(cur_q)
        for _ in range(120):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc, vc = rng.randrange(2), rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = FULL if vc else 0
            for b_i in _iter_bits(vm):
                rv ^= cur_v[b_i]
            prod = lv & rv
            if _add_p(dict(w_pivs), prod):
                _add_p(w_pivs, prod)
                apply_and_xor(builder, cur_q, q, um, vm, uc, vc)
                ccx_total += 1
                cur_q.append(q)
                cur_v.append(prod)
                if is_tmp:
                    tmp_vals.append(prod)
                break
        else:
            raise RuntimeError('Failed to find independent bootstrap wire in synthesize_qrom1_overwrite')

    # Track planned ops, broadcastable products, and last_read_op per qubit!
    ops_plan: list[dict] = []
    prod_recs: list[tuple[int, int]] = []  # (op_idx, prod_val)
    last_read: dict[km.q, int] = {q: -1 for q in cur_q}

    done_p = dict(base_pivs)
    rem = list(core_ts)
    built_map: dict[km.q, int] = {}

    def _record_step(q_tgt: km.q, src_q: list[km.q], src_v: list[int],
                     safe_op_idxs: list[int], bm: int, p_uv: list[tuple[int, int]],
                     lm: int, cb: int):
        for pi in _iter_bits(bm):
            ops_plan[safe_op_idxs[pi]]['targets'].append(q_tgt)
        snap_q = list(src_q)
        for um, vm in p_uv:
            lv = 0
            for a in _iter_bits(um):
                lv ^= src_v[a]
            rv = 0
            for b_i in _iter_bits(vm):
                rv ^= src_v[b_i]
            op_idx = len(ops_plan)
            ops_plan.append({'type': 'ccx', 'wires': snap_q, 'targets': [q_tgt], 'm1': um, 'm2': vm})
            prod_recs.append((op_idx, lv & rv))
            for a in _iter_bits(um | vm):
                last_read[snap_q[a]] = op_idx
        op_idx = len(ops_plan)
        ops_plan.append({'type': 'lin', 'wires': snap_q, 'target': q_tgt, 'lin_mask': lm, 'const_bit': cb})
        for a in _iter_bits(lm):
            last_read[snap_q[a]] = op_idx

    def _consume_goal(goal: int):
        nonlocal rem
        _add_p(done_p, goal)
        new_rem = []
        tp2 = dict(done_p)
        for tv in rem:
            if _add_p(tp2, tv):
                new_rem.append(tv)
        rem = new_rem

    # Phase 2: Fresh outputs `core_qs[n_boot:]` (all earlier ops safe to broadcast!)
    hold_set_init = set(core_qs[:min(k_hold, n_boot)])
    for q in core_qs[n_boot:]:
        if hold_set_init:
            src_idx = [i for i, w in enumerate(cur_q) if w not in hold_set_init]
        else:
            src_idx = list(range(len(cur_q)))
        src_q = [cur_q[i] for i in src_idx]
        src_v = [cur_v[i] for i in src_idx]
        safe_ops = [oi for oi, _ in prod_recs]
        safe_pv = [pv for _, pv in prod_recs]
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(src_v, safe_pv, rem, n_in=n_in, rng=rng, cur_val=0)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= safe_pv[pi]
        assert (_verify_quad_eval(src_v, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        _record_step(q, src_q, src_v, safe_ops, bm, p_uv, lm, cb)
        cur_q.append(q)
        cur_v.append(goal)
        last_read[q] = -1
        built_map[q] = goal
        _consume_goal(goal)

    # Phase 3: In-place overwrite `core_qs[:n_boot]`!
    # Always pick the pending bootstrap qubit with smallest `last_read[q]` (maximizing safe broadcast products!),
    # and optionally exclude the next `k_hold` pending qubits from `src_q` so they accumulate broadcast products!
    pending_boot = list(core_qs[:n_boot])
    while pending_boot:
        # Sort pending by last_read so the one with most safe broadcast products goes first
        pending_boot.sort(key=lambda w: last_read[w])
        q = pending_boot.pop(0)
        pos = cur_q.index(q)
        old_val = cur_v[pos]
        excl = {q}
        if k_hold > 0 and pending_boot:
            excl.update(pending_boot[:min(k_hold, len(pending_boot))])
        src_idx = [i for i, w in enumerate(cur_q) if w not in excl]
        src_q = [cur_q[i] for i in src_idx]
        src_v = [cur_v[i] for i in src_idx]
        lr = last_read[q]
        safe_ops = [oi for oi, _ in prod_recs if oi > lr]
        safe_pv = [pv for oi, pv in prod_recs if oi > lr]
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(src_v, safe_pv, rem, n_in=n_in, rng=rng, cur_val=old_val)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= safe_pv[pi]
        assert (old_val ^ _verify_quad_eval(src_v, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        _record_step(q, src_q, src_v, safe_ops, bm, p_uv, lm, cb)
        cur_v[pos] = goal
        last_read[q] = len(ops_plan) - 1  # cannot retroactively broadcast past its own overwrite
        built_map[q] = goal
        _consume_goal(goal)

    # Emit planned ops into builder!
    for op in ops_plan:
        if op['type'] == 'ccx':
            ccx_total += apply_and_xor_bcast(builder, op['wires'], op['targets'], op['m1'], op['m2'])
        else:
            t = op['target']
            for i in _iter_bits(op['lin_mask']):
                builder.cx(op['wires'][i], t)
            if op['const_bit']:
                builder.x(t)

    # Phase 4: Free `temp_qubits` via 0-CCX HMR+CZ, linear cleanup -> core_ts, fill dep_idx!
    core_built = [built_map[q] for q in core_qs]
    if temp_qubits:
        uncompute_bits_hmr(
            builder, list(addr_qubits) + core_qs, list(a_vals) + core_built, temp_qubits, tmp_vals, n_in=n_in,
        )

    _gf2_linear_transform_inplace(builder, list(addr_qubits), list(a_vals), core_qs, core_built, core_ts, n_in=n_in)

    if dep_idx:
        all_q = list(addr_qubits) + core_qs
        all_v = list(a_vals) + core_ts
        dp: dict[int, int] = {}
        dc: dict[int, int] = {}

        def _ins_d(v, idx):
            comb = 1 << idx
            while v:
                lsb = v & -v
                if lsb in dp:
                    v ^= dp[lsb]
                    comb ^= dc[lsb]
                else:
                    dp[lsb] = v
                    dc[lsb] = comb
                    return

        _ins_d(FULL, 0)
        for i, v in enumerate(all_v):
            _ins_d(v, 1 + i)
        for j in dep_idx:
            t = target_vals[j]
            tc = 0
            while t:
                lsb = t & -t
                t ^= dp[lsb]
                tc ^= dc[lsb]
            if tc & 1:
                builder.x(out_qubits[j])
            for k in _iter_bits(tc >> 1):
                builder.cx(all_q[k], out_qubits[j])

    return ccx_total


def compute_quad_bits_overwrite(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    W_vals: Sequence[int],
    new_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    n_in: int = 256,
    k_fresh: int = 1,
    k_hold: int = 0,
    seed: int = 0,
    use_init_and_temp: bool = False,
) -> int:
    """v34-style Boot + In-Place Overwrite + Safe Broadcast for mid/post LUTs (`c2_fwd`, `c0_lo_rest`, `c2_rev`, `addr_rev`)!"""
    FULL = (1 << n_in) - 1
    rng = random.Random(seed)
    cur_q = list(W_qubits)
    cur_v = list(W_vals)
    ccx_total = 0

    pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(pivs, FULL)
    for v in cur_v:
        _add_p(pivs, v)

    indep_idx: list[int] = []
    dep_idx: list[int] = []
    tp_chk = dict(pivs)
    for j, tv in enumerate(target_vals):
        if _add_p(tp_chk, tv):
            indep_idx.append(j)
        else:
            dep_idx.append(j)

    core_qs = [new_qubits[j] for j in indep_idx]
    core_ts = [target_vals[j] for j in indep_idx]
    m_indep = len(core_qs)
    if m_indep <= 1:
        return compute_quad_bits_bcast(
            builder, W_qubits, W_vals, new_qubits, target_vals,
            temp_qubits=temp_qubits, n_in=n_in, seed=seed, use_init_and_temp=use_init_and_temp,
        )

    k_fresh = max(1, min(k_fresh, m_indep))
    n_boot = m_indep - k_fresh

    # Phase 1: 1-CCX boot `temp_qubits` + `core_qs[:n_boot]`!
    tmp_vals: list[int] = []
    tmp_init_recs: list[tuple[km.q, int, int, int, int, int]] = []
    for q, is_tmp in [(tq, True) for tq in temp_qubits] + [(bq, False) for bq in core_qs[:n_boot]]:
        M_cur = len(cur_q)
        for _ in range(120):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc, vc = rng.randrange(2), rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = FULL if vc else 0
            for b_i in _iter_bits(vm):
                rv ^= cur_v[b_i]
            prod = lv & rv
            if _add_p(dict(pivs), prod):
                _add_p(pivs, prod)
                if is_tmp and use_init_and_temp:
                    _apply_init_and_linear(builder, cur_q, q, um, vm, uc, vc, uncompute=False)
                    tmp_init_recs.append((q, M_cur, um, vm, uc, vc))
                else:
                    apply_and_xor(builder, cur_q, q, um, vm, uc, vc)
                ccx_total += 1
                cur_q.append(q)
                cur_v.append(prod)
                if is_tmp:
                    tmp_vals.append(prod)
                break

    ops_plan: list[dict] = []
    prod_recs: list[tuple[int, int]] = []
    last_read: dict[km.q, int] = {q: -1 for q in cur_q}
    done_p = dict(pivs)  # wait: done_p should only contain W_vals + FULL, not boot vals!
    done_p = {}
    _add_p(done_p, FULL)
    for v in W_vals:
        _add_p(done_p, v)
    rem = list(core_ts)
    built_map: dict[km.q, int] = {}

    def _record_step(q_tgt, src_q, src_v, safe_ops, bm, p_uv, lm, cb):
        for pi in _iter_bits(bm):
            ops_plan[safe_ops[pi]]['targets'].append(q_tgt)
        snap_q = list(src_q)
        for um, vm in p_uv:
            lv = 0
            for a in _iter_bits(um):
                lv ^= src_v[a]
            rv = 0
            for b_i in _iter_bits(vm):
                rv ^= src_v[b_i]
            op_idx = len(ops_plan)
            ops_plan.append({'type': 'ccx', 'wires': snap_q, 'targets': [q_tgt], 'm1': um, 'm2': vm})
            prod_recs.append((op_idx, lv & rv))
            for a in _iter_bits(um | vm):
                last_read[snap_q[a]] = op_idx
        op_idx = len(ops_plan)
        ops_plan.append({'type': 'lin', 'wires': snap_q, 'target': q_tgt, 'lin_mask': lm, 'const_bit': cb})
        for a in _iter_bits(lm):
            last_read[snap_q[a]] = op_idx

    def _consume_goal(goal: int):
        nonlocal rem
        _add_p(done_p, goal)
        new_rem = []
        tp2 = dict(done_p)
        for tv in rem:
            if _add_p(tp2, tv):
                new_rem.append(tv)
        rem = new_rem

    hold_init = set(core_qs[:min(k_hold, n_boot)])
    for q in core_qs[n_boot:]:
        src_idx = [i for i, w in enumerate(cur_q) if w not in hold_init] if hold_init else list(range(len(cur_q)))
        src_q = [cur_q[i] for i in src_idx]
        src_v = [cur_v[i] for i in src_idx]
        safe_ops = [oi for oi, _ in prod_recs]
        safe_pv = [pv for _, pv in prod_recs]
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(src_v, safe_pv, rem, n_in=n_in, rng=rng, cur_val=0)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= safe_pv[pi]
        assert (_verify_quad_eval(src_v, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        _record_step(q, src_q, src_v, safe_ops, bm, p_uv, lm, cb)
        cur_q.append(q)
        cur_v.append(goal)
        last_read[q] = -1
        built_map[q] = goal
        _consume_goal(goal)

    pending_boot = list(core_qs[:n_boot])
    while pending_boot:
        pending_boot.sort(key=lambda w: last_read[w])
        q = pending_boot.pop(0)
        pos = cur_q.index(q)
        old_val = cur_v[pos]
        excl = {q}
        if k_hold > 0 and pending_boot:
            excl.update(pending_boot[:min(k_hold, len(pending_boot))])
        src_idx = [i for i, w in enumerate(cur_q) if w not in excl]
        src_q = [cur_q[i] for i in src_idx]
        src_v = [cur_v[i] for i in src_idx]
        lr = last_read[q]
        safe_ops = [oi for oi, _ in prod_recs if oi > lr]
        safe_pv = [pv for oi, pv in prod_recs if oi > lr]
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(src_v, safe_pv, rem, n_in=n_in, rng=rng, cur_val=old_val)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= safe_pv[pi]
        assert (old_val ^ _verify_quad_eval(src_v, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        _record_step(q, src_q, src_v, safe_ops, bm, p_uv, lm, cb)
        cur_v[pos] = goal
        last_read[q] = len(ops_plan) - 1
        built_map[q] = goal
        _consume_goal(goal)

    for op in ops_plan:
        if op['type'] == 'ccx':
            ccx_total += apply_and_xor_bcast(builder, op['wires'], op['targets'], op['m1'], op['m2'])
        else:
            t = op['target']
            for i in _iter_bits(op['lin_mask']):
                builder.cx(op['wires'][i], t)
            if op['const_bit']:
                builder.x(t)

    core_built = [built_map[q] for q in core_qs]
    _gf2_linear_transform_inplace(builder, list(W_qubits), list(W_vals), core_qs, core_built, core_ts, n_in=n_in)

    if dep_idx:
        all_q = list(W_qubits) + core_qs
        all_v = list(W_vals) + core_ts
        dp: dict[int, int] = {}
        dc: dict[int, int] = {}

        def _ins_d(v, idx):
            comb = 1 << idx
            while v:
                lsb = v & -v
                if lsb in dp:
                    v ^= dp[lsb]
                    comb ^= dc[lsb]
                else:
                    dp[lsb] = v
                    dc[lsb] = comb
                    return

        _ins_d(FULL, 0)
        for i, v in enumerate(all_v):
            _ins_d(v, 1 + i)
        for j in dep_idx:
            t = target_vals[j]
            tc = 0
            while t:
                lsb = t & -t
                t ^= dp[lsb]
                tc ^= dc[lsb]
            if tc & 1:
                builder.x(new_qubits[j])
            for k in _iter_bits(tc >> 1):
                builder.cx(all_q[k], new_qubits[j])

    if temp_qubits:
        if use_init_and_temp:
            for tq, M_prev, um, vm, uc, vc in reversed(tmp_init_recs):
                _apply_init_and_linear(builder, cur_q[:M_prev], tq, um, vm, uc, vc, uncompute=True)
        else:
            keep_q = list(W_qubits) + list(new_qubits)
            keep_v = list(W_vals) + list(target_vals)
            uncompute_bits_hmr(builder, keep_q, keep_v, temp_qubits, tmp_vals, n_in=n_in)

    return ccx_total


def synthesize_qrom1_bcast(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    out_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    n_boot: int = 12,
    seed: int = 0,
) -> int:
    """Two-phase zero-ancilla QROM1 using shared-product CCX broadcasting + 0-Tof HMR boot free!

    Phase 1: Bootstrap `n_boot` of `core_qubits` + `temp_qubits` with 1 CCX each (M1 ~ 25).
    Phase 2: Compute `core_qubits[n_boot:]` via `compute_quad_bits_bcast` (1 massive BCAST wave!).
    Phase 3: Free `core_qubits[:n_boot]` back to |0> with 0 CCX via `HMR+CZ`!
    Phase 4: Compute `core_qubits[:n_boot]` via `compute_quad_bits_bcast` from `addr + Phase2 + temp`!
    Phase 5: Free `temp_qubits` (0 CCX HMR+CZ), linear cleanup (0 CCX), fill dependent bits (0 CCX).
    """
    n_addr = len(addr_qubits)
    n_in = 1 << n_addr
    FULL = (1 << n_in) - 1
    rng = random.Random(seed)
    a_vals = addr_bit_masks(n_addr)

    base_pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(base_pivs, FULL)
    for v in a_vals:
        _add_p(base_pivs, v)

    indep_idx: list[int] = []
    dep_idx: list[int] = []
    t_pivs = dict(base_pivs)
    for j, tv in enumerate(target_vals):
        if _add_p(t_pivs, tv):
            indep_idx.append(j)
        else:
            dep_idx.append(j)

    core_qs = [out_qubits[j] for j in indep_idx]
    core_ts = [target_vals[j] for j in indep_idx]
    m_indep = len(core_qs)
    n_tmp = len(temp_qubits)

    # Ensure M1 = n_addr + n_boot + n_tmp >= 25 (guaranteed 256/256 quadratic span!)
    min_boot = max(0, 25 - n_addr - n_tmp)
    n_boot = min(m_indep, max(n_boot, min_boot))
    if m_indep - n_boot < 15:
        # Fallback to standard synthesize_qrom1 if table is small
        return synthesize_qrom1(builder, addr_qubits, out_qubits, target_vals, temp_qubits=temp_qubits, seed=seed)

    # Phase 1: Bootstrap `temp_qubits` + `core_qs[:n_boot]` with 1 CCX each (greedily ensuring full quad span!)
    ccx_total = 0
    W_q: list[km.q] = list(addr_qubits)
    W_v: list[int] = list(a_vals)
    w_pivs = dict(base_pivs)
    tmp_vals: list[int] = []
    boot_vals: list[int] = []

    for q, is_tmp in [(tq, True) for tq in temp_qubits] + [(bq, False) for bq in core_qs[:n_boot]]:
        M_cur = len(W_q)
        for _ in range(100):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc, vc = rng.randrange(2), rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= W_v[a]
            rv = FULL if vc else 0
            for b in _iter_bits(vm):
                rv ^= W_v[b]
            prod = lv & rv
            if _add_p(dict(w_pivs), prod):
                _add_p(w_pivs, prod)
                apply_and_xor(builder, W_q, q, um, vm, uc, vc)
                ccx_total += 1
                W_q.append(q)
                W_v.append(prod)
                if is_tmp:
                    tmp_vals.append(prod)
                else:
                    boot_vals.append(prod)
                break
        else:
            raise RuntimeError('Failed to find independent bootstrap wire in synthesize_qrom1_bcast')

    # Phase 2: Compute `core_qs[n_boot:]` via BCAST + multi-target!
    p2_qs = core_qs[n_boot:]
    done_p = dict(base_pivs)
    rem = list(core_ts)
    p2_built: list[int] = []
    ops_p2: list[dict] = []
    prods_p2: list[int] = []
    pop_idx_p2: list[int] = []

    cur_q = list(W_q)
    cur_v = list(W_v)
    for q in p2_qs:
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(cur_v, prods_p2, rem, n_in=n_in, rng=rng)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            ops_p2[pop_idx_p2[pi]]['targets'].append(q)
            bcast_val ^= prods_p2[pi]
        assert (_verify_quad_eval(cur_v, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        snap_q = list(cur_q)
        for um, vm in p_uv:
            lv = 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = 0
            for b in _iter_bits(vm):
                rv ^= cur_v[b]
            op_idx = len(ops_p2)
            ops_p2.append({'type': 'ccx', 'wires': snap_q, 'targets': [q], 'm1': um, 'm2': vm})
            prods_p2.append(lv & rv)
            pop_idx_p2.append(op_idx)
        ops_p2.append({'type': 'lin', 'wires': snap_q, 'target': q, 'lin_mask': lm, 'const_bit': cb})
        cur_q.append(q)
        cur_v.append(goal)
        p2_built.append(goal)
        _add_p(done_p, goal)
        new_rem = []
        tp2 = dict(done_p)
        for tv in rem:
            if _add_p(tp2, tv):
                new_rem.append(tv)
        rem = new_rem

    for op in ops_p2:
        if op['type'] == 'ccx':
            ccx_total += apply_and_xor_bcast(builder, op['wires'], op['targets'], op['m1'], op['m2'])
        else:
            t = op['target']
            for i in _iter_bits(op['lin_mask']):
                builder.cx(op['wires'][i], t)
            if op['const_bit']:
                builder.x(t)

    # Phase 3: Free `core_qs[:n_boot]` back to clean |0> with 0 CCX via HMR+CZ!
    keep3_q = list(addr_qubits) + list(temp_qubits) + list(p2_qs)
    keep3_v = list(a_vals) + list(tmp_vals) + list(p2_built)
    uncompute_bits_hmr(builder, keep3_q, keep3_v, core_qs[:n_boot], boot_vals, n_in=n_in)

    # Phase 4: Compute `core_qs[:n_boot]` (fresh |0>!) via BCAST + multi-target from `keep3_q`!
    assert len(rem) == n_boot
    p4_qs = core_qs[:n_boot]
    p4_built: list[int] = []
    ops_p4: list[dict] = []
    prods_p4: list[int] = []
    pop_idx_p4: list[int] = []

    cur_q4 = list(keep3_q)
    cur_v4 = list(keep3_v)
    for q in p4_qs:
        ct, cb, lm, bm, p_uv = _solve_bcast_quad_multi(cur_v4, prods_p4, rem, n_in=n_in, rng=rng)
        goal = 0
        for ti in _iter_bits(ct):
            goal ^= rem[ti]
        bcast_val = 0
        for pi in _iter_bits(bm):
            ops_p4[pop_idx_p4[pi]]['targets'].append(q)
            bcast_val ^= prods_p4[pi]
        assert (_verify_quad_eval(cur_v4, p_uv, lm, cb, n_in) ^ bcast_val) == goal
        snap_q = list(cur_q4)
        for um, vm in p_uv:
            lv = 0
            for a in _iter_bits(um):
                lv ^= cur_v4[a]
            rv = 0
            for b in _iter_bits(vm):
                rv ^= cur_v4[b]
            op_idx = len(ops_p4)
            ops_p4.append({'type': 'ccx', 'wires': snap_q, 'targets': [q], 'm1': um, 'm2': vm})
            prods_p4.append(lv & rv)
            pop_idx_p4.append(op_idx)
        ops_p4.append({'type': 'lin', 'wires': snap_q, 'target': q, 'lin_mask': lm, 'const_bit': cb})
        cur_q4.append(q)
        cur_v4.append(goal)
        p4_built.append(goal)
        _add_p(done_p, goal)
        new_rem = []
        tp2 = dict(done_p)
        for tv in rem:
            if _add_p(tp2, tv):
                new_rem.append(tv)
        rem = new_rem

    assert len(rem) == 0
    for op in ops_p4:
        if op['type'] == 'ccx':
            ccx_total += apply_and_xor_bcast(builder, op['wires'], op['targets'], op['m1'], op['m2'])
        else:
            t = op['target']
            for i in _iter_bits(op['lin_mask']):
                builder.cx(op['wires'][i], t)
            if op['const_bit']:
                builder.x(t)

    # Phase 5: Free `temp_qubits` via 0-Tof HMR+CZ -> exact 0-Tof linear cleanup -> fill `dep_idx`!
    all_core_built = p4_built + p2_built
    if temp_qubits:
        uncompute_bits_hmr(
            builder, list(addr_qubits) + list(core_qs), list(a_vals) + all_core_built,
            temp_qubits, tmp_vals, n_in=n_in,
        )

    _gf2_linear_transform_inplace(builder, addr_qubits, a_vals, core_qs, all_core_built, core_ts, n_in=n_in)

    if dep_idx:
        dp: dict[int, int] = {}
        dc: dict[int, int] = {}

        def _ins_d(v, idx):
            comb = 1 << idx
            while v:
                lsb = v & -v
                if lsb in dp:
                    v ^= dp[lsb]
                    comb ^= dc[lsb]
                else:
                    dp[lsb] = v
                    dc[lsb] = comb
                    return

        _ins_d(FULL, 0)
        for i in range(n_addr):
            _ins_d(a_vals[i], 1 + i)
        for i in range(m_indep):
            _ins_d(core_ts[i], 1 + n_addr + i)
        src_q = list(addr_qubits) + list(core_qs)
        for j in dep_idx:
            t = target_vals[j]
            tc = 0
            while t:
                lsb = t & -t
                t ^= dp[lsb]
                tc ^= dc[lsb]
            if tc & 1:
                builder.x(out_qubits[j])
            for k in _iter_bits(tc >> 1):
                builder.cx(src_q[k], out_qubits[j])

    return ccx_total


def compute_quad_bits(
    builder: km.CircuitBuilder,
    W_qubits: Sequence[km.q],
    W_vals: Sequence[int],
    new_qubits: Sequence[km.q],
    target_vals: Sequence[int],
    temp_qubits: Sequence[km.q] = (),
    n_in: int = 256,
    seed: int = 0,
    use_init_and_temp: bool = False,
    keep_temp_out: dict | None = None,
) -> int:
    """Computes `new_qubits[j] = target_vals[j]` sequentially from `W_qubits` (growing `W`!).

    If `temp_qubits` are provided, first boosts the basis with 1-CCX random products into
    `temp_qubits` (reducing rank `r` of every output bit!), computes `new_qubits` using
    multi-target subspace intersection + linear cleanup if len(new_qubits) >= 3, and finally
    frees `temp_qubits` with 0-Toffoli HMR+CZ (or 0-Toffoli `del_and` when `use_init_and_temp=True`,
    which works even when `len(W_qubits) + len(new_qubits) < 23`!).
    """
    FULL = (1 << n_in) - 1
    rng = random.Random(seed)
    cur_q = list(W_qubits)
    cur_v = list(W_vals)
    ccx_total = 0

    # Optional 1-CCX basis boosting into temp_qubits (freed with 0-Tof HMR+CZ or del_and at end!)
    pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(pivs, FULL)
    for v in cur_v:
        _add_p(pivs, v)

    tmp_vals: list[int] = []
    tmp_init_recs: list[tuple[km.q, int, int, int, int, int]] = []
    for tq in temp_qubits:
        M_cur = len(cur_q)
        for _ in range(100):
            um = rng.randrange(1, 1 << M_cur)
            vm = rng.randrange(1, 1 << M_cur)
            if um == vm:
                continue
            uc, vc = rng.randrange(2), rng.randrange(2)
            lv = FULL if uc else 0
            for a in _iter_bits(um):
                lv ^= cur_v[a]
            rv = FULL if vc else 0
            for b in _iter_bits(vm):
                rv ^= cur_v[b]
            prod = lv & rv
            if _add_p(dict(pivs), prod):
                _add_p(pivs, prod)
                if use_init_and_temp:
                    _apply_init_and_linear(builder, cur_q, tq, um, vm, uc, vc, uncompute=False)
                    tmp_init_recs.append((tq, M_cur, um, vm, uc, vc))
                else:
                    apply_and_xor(builder, cur_q, tq, um, vm, uc, vc)
                ccx_total += 1
                cur_q.append(tq)
                cur_v.append(prod)
                tmp_vals.append(prod)
                break

    # Compute `new_qubits`! Use multi-target intersection when >= 2 linearly independent targets!
    base_pivs = dict(pivs)
    rem = list(target_vals)
    # Check if all target_vals are linearly independent modulo (1, cur_v); if so, multi-target + linear cleanup!
    test_p = dict(base_pivs)
    all_indep = all(_add_p(test_p, tv) for tv in target_vals)
    if all_indep and len(new_qubits) >= 2:
        done_p = dict(base_pivs)
        built_v: list[int] = []
        for q in new_qubits:
            M_av = len(cur_v)
            d = len(rem)
            r_single = 1
            while r_single < M_av and (1 + M_av) + r_single * M_av - r_single * (r_single + 1) // 2 < n_in:
                r_single += 1
            r_multi = 1
            while r_multi < M_av and (1 + M_av) + r_multi * M_av - r_multi * (r_multi + 1) // 2 + d < n_in + 2:
                r_multi += 1
            hit = None
            if r_multi < r_single and d >= 2:
                for r_try in range(r_multi, r_single):
                    hit = _solve_multi_target(cur_v, rem, r_try, n_in, rng, trials=10, cur_val=0)
                    if hit is not None:
                        break
            if hit is not None:
                ct, cb, lm, p_uv = hit
                goal = 0
                for ti in _iter_bits(ct):
                    goal ^= rem[ti]
            else:
                goal = rem[0]
                cb, lm, p_uv = solve_low_rank_quad(cur_v, goal, n_in=n_in, rng=rng, try_r_below=1)
            ccx_total += apply_quad_xor(builder, cur_q, q, p_uv, lm, cb)
            cur_q.append(q)
            cur_v.append(goal)
            built_v.append(goal)
            _add_p(done_p, goal)
            new_rem = []
            tp = dict(done_p)
            for tv in rem:
                if _add_p(tp, tv):
                    new_rem.append(tv)
            rem = new_rem
        # Exact 0-Tof linear cleanup on `new_qubits` using `(W_qubits + temp_qubits)` read-only!
        n_src = len(W_qubits) + len(temp_qubits)
        _gf2_linear_transform_inplace(
            builder, cur_q[:n_src], cur_v[:n_src], new_qubits, built_v, target_vals, n_in=n_in
        )
        cur_v[n_src:] = list(target_vals)
    else:
        for q, tv in zip(new_qubits, target_vals):
            cb, lm, p_uv = solve_low_rank_quad(cur_v, tv, n_in=n_in, rng=rng, try_r_below=1)
            ccx_total += apply_quad_xor(builder, cur_q, q, p_uv, lm, cb)
            cur_q.append(q)
            cur_v.append(tv)

    # Free `temp_qubits` via 0-Toffoli `del_and` (if init_and_temp) or 0-Toffoli HMR+CZ!
    if temp_qubits:
        if use_init_and_temp:
            for tq, M_prev, um, vm, uc, vc in reversed(tmp_init_recs):
                _apply_init_and_linear(builder, cur_q[:M_prev], tq, um, vm, uc, vc, uncompute=True)
        else:
            keep_q = list(W_qubits) + list(new_qubits)
            keep_v = list(W_vals) + list(target_vals)
            uncompute_bits_hmr(builder, keep_q, keep_v, temp_qubits, tmp_vals, n_in=n_in)

    return ccx_total


def uncompute_bits_hmr(
    builder: km.CircuitBuilder,
    keep_qubits: Sequence[km.q],
    keep_vals: Sequence[int],
    drop_qubits: Sequence[km.q],
    drop_vals: Sequence[int],
    n_in: int = 256,
) -> None:
    """Uncomputes ALL `drop_qubits` to |0> using 0 Toffolis via HMR+CZ!

    Uncomputes sequentially (each dropped qubit can use all still-alive drop_qubits + keep_qubits!).
    Raises RuntimeError if `keep_qubits` alone does not span the last dropped qubit quadratically.
    """
    rem_q = list(keep_qubits) + list(drop_qubits)
    rem_v = list(keep_vals) + list(drop_vals)
    for _ in range(len(drop_qubits)):
        q = rem_q.pop()
        tv = rem_v.pop()
        ok, cb, lm, qp = find_any_quad(rem_v, tv, n_in=n_in)
        if not ok:
            raise RuntimeError(f'Cannot 0-Toffoli HMR uncompute qubit with {len(rem_v)} remaining wires!')
        uncompute_quad_hmr(builder, rem_q, q, qp, lm, cb)


def hmr_defer_bits(
    builder: km.CircuitBuilder,
    drop_qubits: Sequence[km.q],
) -> list[km.b]:
    """Immediately frees `drop_qubits` via HMR into classical bits (0 Toffolis!),
    deferring the diagonal CZ/Z phase fixup until `apply_deferred_quad_fixup`."""
    bits: list[km.b] = []
    for q in drop_qubits:
        mb = builder.alloc_bits(1)[0]
        builder.hmr(q, mb)
        bits.append(mb)
    return bits


def apply_deferred_quad_fixup(
    builder: km.CircuitBuilder,
    basis_qubits: Sequence[km.q],
    basis_vals: Sequence[int],
    target_vals: Sequence[int],
    hmr_bits: Sequence[km.b],
    n_in: int = 256,
) -> None:
    """Applies 0-Toffoli diagonal CZ/Z/NEG phase fixups for previously deferred HMR bits
    once `basis_qubits` hold pure functions spanning `target_vals` quadratically, then frees `hmr_bits`."""
    assert len(target_vals) == len(hmr_bits)
    for tv, mb in zip(target_vals, hmr_bits):
        ok, cb, lm, qp = find_any_quad(basis_vals, tv, n_in=n_in)
        if not ok:
            raise RuntimeError(f'Deferred HMR target not in quadratic span of {len(basis_vals)} wires!')
        builder.push_condition(mb)
        if cb:
            builder.neg()
        for i in _iter_bits(lm):
            builder.z(basis_qubits[i])
        for i, j in qp:
            builder.cz(basis_qubits[i], basis_qubits[j])
        builder.pop_condition()
        builder.free(mb)


def _apply_init_and_linear(
    builder: km.CircuitBuilder,
    wires: Sequence[km.q],
    target: km.q,
    mask1: int,
    mask2: int,
    c1: int,
    c2: int,
    *,
    uncompute: bool = False,
) -> None:
    """Computes `target = L1 & L2` onto clean |0> via `init_and` (1 CCX), or uncomputes via `del_and` (0 CCX)."""
    b1 = _iter_bits(mask1)
    b2 = _iter_bits(mask2)
    assert b1 and b2
    if mask1 == mask2:
        # Degenerate shouldn't happen for nonlinear product
        raise ValueError('mask1 == mask2 in _apply_init_and_linear')
    p1 = b1[0]
    p2 = b2[0]
    if p1 == p2:
        if len(b2) > 1:
            p2 = b2[1]
        elif len(b1) > 1:
            p1 = b1[1]
        else:
            raise ValueError('Single identical wire in _apply_init_and_linear')

    # Compute L1 onto wires[p1], then L2 onto wires[p2] (handling p1 in b2 safely)
    p1_in_b2 = (mask2 >> p1) & 1
    p2_in_b1 = (mask1 >> p2) & 1
    assert not (p1_in_b2 and p2_in_b1) or len(b1) + len(b2) > 2
    # Simpler exact handling matching apply_and_xor:
    # Build L1 on p1 (excluding p2 if p2 in b1 -> do p2 first? Check!)
    if p1_in_b2 and p2_in_b1:
        # Pick distinct pivots where at least one is not in the other mask (always exists since mask1 != mask2!)
        diff1 = mask1 & ~mask2
        diff2 = mask2 & ~mask1
        if diff1:
            p1 = _iter_bits(diff1)[0]
            p2 = b2[0]
        else:
            p2 = _iter_bits(diff2)[0]
            p1 = b1[0]
        p1_in_b2 = (mask2 >> p1) & 1
        p2_in_b1 = (mask1 >> p2) & 1

    # Order: modify the pivot that does NOT appear in the other mask SECOND!
    def _prep_first_second(pa, ba, ca, pb, bb, cb):
        # pa may appear in bb, pb does NOT appear in ba!
        # Modify pb first (using unmodified pa if needed), then modify pa!
        for j in bb:
            if j != pb:
                builder.cx(wires[j], wires[pb])
        if cb:
            builder.x(wires[pb])
        for i in ba:
            if i != pa:
                builder.cx(wires[i], wires[pa])
        if ca:
            builder.x(wires[pa])

    def _undo_first_second(pa, ba, ca, pb, bb, cb):
        if ca:
            builder.x(wires[pa])
        for i in reversed(ba):
            if i != pa:
                builder.cx(wires[i], wires[pa])
        if cb:
            builder.x(wires[pb])
        for j in reversed(bb):
            if j != pb:
                builder.cx(wires[j], wires[pb])

    if not p2_in_b1:
        _prep_first_second(p1, b1, c1, p2, b2, c2)
        if not uncompute:
            builder.init_and(wires[p1], wires[p2], target)
        else:
            builder.del_and(wires[p1], wires[p2], target)
        _undo_first_second(p1, b1, c1, p2, b2, c2)
    else:
        _prep_first_second(p2, b2, c2, p1, b1, c1)
        if not uncompute:
            builder.init_and(wires[p1], wires[p2], target)
        else:
            builder.del_and(wires[p1], wires[p2], target)
        _undo_first_second(p2, b2, c2, p1, b1, c1)


def uncompute_qrom1_init_and(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    out_qubits: Sequence[km.q],
    out_vals: Sequence[int],
    alloc=None,
    seed: int = 0,
) -> tuple[int, int]:
    """Uncomputes ALL `out_qubits` (pure functions of `addr_qubits`) using greedy `init_and`
    basis wires + 0-Tof HMR+CZ + 0-Tof `del_and` (~15-17 CCX total, beating 25-CCX `del_lookup`!).

    Returns (n_hmr_free, n_init_and_ccx).
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))

    n_addr = len(addr_qubits)
    n_in = 1 << n_addr
    FULL = (1 << n_in) - 1
    a_vals = addr_bit_masks(n_addr)
    rem_q = list(out_qubits)
    rem_v = list(out_vals)
    n_hmr_free = 0

    # Step 1: Peel off as many out_qubits as possible with 0 CCX!
    while rem_q:
        cand_q = list(addr_qubits) + rem_q[:-1]
        cand_v = a_vals + rem_v[:-1]
        ok, cb, lm, qp = find_any_quad(cand_v, rem_v[-1], n_in=n_in)
        if ok:
            q_drop = rem_q.pop()
            rem_v.pop()
            uncompute_quad_hmr(builder, cand_q, q_drop, qp, lm, cb)
            n_hmr_free += 1
        else:
            break

    if not rem_q:
        return n_hmr_free, 0

    # Step 2: Greedily build `init_and` temp basis from `addr_qubits` until ALL `rem_v` are spanned!
    rng = random.Random(seed)
    ws_q: list[km.q] = list(addr_qubits)
    ws_v: list[int] = list(a_vals)
    pivs: dict[int, int] = {}

    def _add_p(p, v):
        while v:
            lsb = v & -v
            if lsb in p:
                v ^= p[lsb]
            else:
                p[lsb] = v
                return True
        return False

    _add_p(pivs, FULL)
    for v in ws_v:
        _add_p(pivs, v)
    for i in range(len(ws_v)):
        for j in range(i + 1, len(ws_v)):
            _add_p(pivs, ws_v[i] & ws_v[j])

    def _all_rem_spanned() -> bool:
        for tv in rem_v:
            v = tv
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                else:
                    return False
        return True

    tmp_records: list[tuple[km.q, int, int, int, int, int]] = []  # (tq, M_prev, m1, m2, c1, c2)
    while not _all_rem_spanned():
        M = len(ws_v)
        best_gain = -1
        best_cand = None
        for _ in range(48):
            m1 = rng.randrange(1, 1 << M)
            m2 = rng.randrange(1, 1 << M)
            if m1 == m2:
                continue
            c1, c2 = rng.randrange(2), rng.randrange(2)
            v1 = FULL if c1 else 0
            for a in _iter_bits(m1):
                v1 ^= ws_v[a]
            v2 = FULL if c2 else 0
            for b in _iter_bits(m2):
                v2 ^= ws_v[b]
            cand = v1 & v2
            tmp_p = dict(pivs)
            g = 0
            for x in [cand] + [cand & w for w in ws_v]:
                if _add_p(tmp_p, x):
                    g += 1
            # Bonus if all rem_v become spanned immediately!
            if g > best_gain:
                best_gain = g
                best_cand = (cand, m1, m2, c1, c2)
                if len(pivs) + g == n_in:
                    break
        assert best_cand is not None
        cand, m1, m2, c1, c2 = best_cand
        tq = _alloc(1)[0]
        _apply_init_and_linear(builder, ws_q, tq, m1, m2, c1, c2, uncompute=False)
        tmp_records.append((tq, M, m1, m2, c1, c2))
        ws_q.append(tq)
        ws_v.append(cand)
        _add_p(pivs, cand)
        for w in ws_v[:-1]:
            _add_p(pivs, cand & w)

    # Step 3: HMR+CZ uncompute ALL remaining `rem_q` using `ws_q` (0 CCX!)
    for q_drop, tv in zip(rem_q, rem_v):
        ok, cb, lm, qp = find_any_quad(ws_v, tv, n_in=n_in)
        assert ok, 'Failed to span rem_v after init_and basis build!'
        uncompute_quad_hmr(builder, ws_q, q_drop, qp, lm, cb)

    # Step 4: Uncompute all `init_and` temp wires in reverse with 0 CCX (`del_and`)!
    for tq, M_prev, m1, m2, c1, c2 in reversed(tmp_records):
        _apply_init_and_linear(builder, ws_q[:M_prev], tq, m1, m2, c1, c2, uncompute=True)
        _free(tq)

    return n_hmr_free, len(tmp_records)


def uncompute_qrom1_hybrid(
    builder: km.CircuitBuilder,
    addr_qubits: Sequence[km.q],
    out_qubits: Sequence[km.q],
    out_vals: Sequence[int],
) -> tuple[int, int]:
    """Uncomputes `out_qubits` (pure functions of `addr_qubits`) to |0> with minimal Toffolis!

    Step 1: Greedily uncomputes qubits with 0-Toffoli HMR+CZ as long as `{addr} U remaining`
            spans the dropped qubit quadratically (typically uncomputes all but ~14-15 qubits!).
    Step 2: Uncomputes the remaining ~14-15 qubits in ONE single `builder.del_lookup`
            (costs only 25 Toffolis total for k=8, 4 temp ancillas!).

    Returns (n_hmr_zero_tof, n_del_lookup_bits).
    """
    n_addr = len(addr_qubits)
    n_in = 1 << n_addr
    a_vals = addr_bit_masks(n_addr)
    rem_q = list(out_qubits)
    rem_v = list(out_vals)
    n_hmr = 0

    while rem_q:
        # Try dropping the last qubit using {addr} + rem[:-1]
        cand_q = list(addr_qubits) + rem_q[:-1]
        cand_v = a_vals + rem_v[:-1]
        ok, cb, lm, qp = find_any_quad(cand_v, rem_v[-1], n_in=n_in)
        if ok:
            q_drop = rem_q.pop()
            rem_v.pop()
            uncompute_quad_hmr(builder, cand_q, q_drop, qp, lm, cb)
            n_hmr += 1
        else:
            break

    if rem_q:
        # Build integer table for remaining bits and call del_lookup!
        rem_tab = [0] * n_in
        for bi, mask in enumerate(rem_v):
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                rem_tab[idx] |= (1 << bi)
                m ^= lsb
        builder.del_lookup(table=rem_tab, address=km.array(list(addr_qubits)), target=km.array(rem_q))

    return n_hmr, len(rem_q)


if __name__ == '__main__':
    import time
    import numpy as np

    print('Unit-testing gf2_lut.py on a 35-bit k=8 table (forward + HMR addr free/restore + hybrid uncompute)...')
    rng = random.Random(123)
    m_out = 35
    tab_vals = [rng.getrandbits(256) for _ in range(m_out)]
    c2_vals = [rng.getrandbits(256) for _ in range(8)]

    t0 = time.time()
    b = km.CircuitBuilder()
    addr = b.create_quantum_register(8, name='x')
    out = b.alloc_qubits(m_out)
    tmp = b.alloc_qubits(2)
    ccx1 = synthesize_qrom1(b, addr, out, tab_vals, temp_qubits=tmp, seed=7)
    b.free(tmp)

    # Free addr (8 qubits) via 0-Tof HMR+CZ (do NOT call b.free(addr) since 'x' is a named register;
    # instead reuse the clean |0> qubit handles addr[0..7] directly as scratch!).
    a_vals = addr_bit_masks(8)
    uncompute_bits_hmr(b, out, tab_vals, addr, a_vals)
    # Let's test computing c2 into addr[0..7] while addr is clean |0>, phase-testing, uncomputing c2, restoring addr!
    ccx_c2 = compute_quad_bits(b, out, tab_vals, addr, c2_vals, seed=11)
    # Apply phase kickback from c2[0] and out[0] to verify exact bit values AND zero spurious phase!
    b.z_pow(addr[0], 0.125)
    b.z_pow(out[0], 0.0625)
    # Uncompute c2 from addr[0..7] with 0-Tof HMR+CZ!
    uncompute_bits_hmr(b, out, tab_vals, addr, c2_vals)
    # Restore addr[0..7] from out!
    ccx_rest = compute_quad_bits(b, out, tab_vals, addr, a_vals, seed=19)
    # Uncompute out via hybrid HMR + del_lookup!
    n_hmr, n_dl = uncompute_qrom1_hybrid(b, addr, out, tab_vals)
    b.free(out)
    circ = b.finish_circuit()
    dt = time.time() - t0
    print(f'Built in {dt:.2f}s: Q={circ.num_qubits}, ccx_qrom1={ccx1}, ccx_c2={ccx_c2}, ccx_rest={ccx_rest}, hmr_uncomp={n_hmr}, del_lookup_bits={n_dl}')
    print('Op stats:', {k: v for k, v in circ.operation_statistics().items() if k in ('CCX', 'CSWAP', 'CCZ', 'HMR', 'CZ', 'Z_POW')})

    # Verify across ALL 256 inputs x 64 shots using kickmix simulator!
    x_reg = km.array([km.q(i) for i in range(8)])
    scratch = km.array([km.q(i) for i in range(8, circ.num_qubits)])
    sim = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=1234)
    xs_u64 = np.arange(256, dtype=np.uint64)
    x_bits = ((xs_u64[None, :] >> np.arange(8, dtype=np.uint64)[:, None]) & 1).astype(np.bool_)
    exp_turns = np.array([
        ((((c2_vals[0] >> x) & 1) * 0.125 + ((tab_vals[0] >> x) & 1) * 0.0625) / 2.0) % 1.0
        for x in range(256)
    ])
    max_diff = 0.0
    for shot in range(64):
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, x_bits)
        sim.do(circ)
        out_bits = sim.read_across_shots(x_reg)
        assert np.array_equal(out_bits, x_bits), f'x not restored on shot {shot}!'
        dirty = sim.read_across_shots(scratch).any(axis=1)
        assert not dirty.any(), f'Scratch qubit {scratch[int(np.argmax(dirty))]} dirty on shot {shot}!'
        # read_shot_phase returns half-turns Fraction -> divide by 2 for turns!
        got_turns = np.array([float(sim.read_shot_phase(i)) / 2.0 for i in range(256)]) % 1.0
        diff = np.abs((got_turns - exp_turns + 0.5) % 1.0 - 0.5)
        max_diff = max(max_diff, float(diff.max()))
    st = sim.operation_statistics()
    mean_tof = (st.get('CCX', 0) + st.get('CSWAP', 0) + st.get('CCZ', 0)) / (256 * 64)
    print(f'Verified {256*64} shots (256 x 64): max phase diff = {max_diff:.2e}, mean_Tof={mean_tof:.2f} -> {"PASS" if max_diff < 1e-12 else "FAIL"}!')
