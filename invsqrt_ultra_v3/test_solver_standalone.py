#!/usr/bin/env python3
"""Standalone GF(2) solver test — NO kickmix, NO imports from gf2_lut.py.
All needed math functions are embedded. Tests the PATCHED gf2_lut.py solver.

Usage:  python test_solver_standalone.py
"""
import random, time, os, sys

# ===== Embedded pure-math functions from gf2_lut.py =====

_HAS_NUMBA = False  # Skip Numba for this test

def _iter_bits(mask):
    out = []
    m = mask
    while m:
        lsb = m & -m
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out

def symplectic_decompose(mat_rows, M):
    A = list(mat_rows)
    pairs_uv = []
    diag_fix = 0
    for i in range(M):
        while A[i]:
            lsb = A[i] & -A[i]
            j = lsb.bit_length() - 1
            u = A[i]
            v = A[j]
            pairs_uv.append((u, v))
            diag_fix ^= (u & v)
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

def _extract_quad_solution(comb, M, u_masks):
    const_bit = comb & 1
    lin_mask = (comb >> 1) & ((1 << M) - 1)
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

def _verify_quad_eval(W_vals, pairs_uv, lin_mask, const_bit, n_in=256):
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

def solve_low_rank_quad(W_vals, target_val, n_in=256, rng=None, try_r_below=1):
    if rng is None:
        rng = random.Random(0)
    FULL = (1 << n_in) - 1
    M = len(W_vals)
    lin_pivs = {}
    lin_c = {}
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
    t = target_val; tc = 0
    while t:
        lsb = t & -t
        if lsb in lin_pivs:
            t ^= lin_pivs[lsb]; tc ^= lin_c[lsb]
        else:
            break
    if t == 0:
        return tc & 1, (tc >> 1) & ((1 << M) - 1), []
    lin_rk = len(lin_pivs)
    r_full = 1
    while r_full < M and lin_rk + r_full * M - r_full * (r_full + 1) // 2 < n_in:
        r_full += 1
    best = None
    for r_try in range(max(1, r_full - try_r_below), r_full + 2):
        n_trials = 6 if r_try == r_full else (3 if r_try < r_full else 10)
        for _ in range(n_trials):
            u_masks = []; u_vecs = []
            for _k in range(r_try):
                um = rng.randrange(1, 1 << M)
                uv = 0
                for a in _iter_bits(um): uv ^= W_vals[a]
                u_masks.append(um); u_vecs.append(uv)
            pivs = dict(lin_pivs); piv_c = dict(lin_c)
            idx = 1 + M
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx); idx += 1
            t = target_val; tc = 0
            while t:
                lsb = t & -t
                if lsb in pivs:
                    t ^= pivs[lsb]; tc ^= piv_c[lsb]
                else:
                    break
            if t == 0:
                cb, lm, p_uv = _extract_quad_solution(tc, M, u_masks)
                if best is None or len(p_uv) < len(best[2]):
                    best = (cb, lm, p_uv)
                    if len(p_uv) <= r_full - 1:
                        return best
        if best is not None:
            return best
    raise RuntimeError(f'solve_low_rank_quad failed for M={M}')


# ===== The PATCHED _solve_bcast_quad_multi =====

def _solve_bcast_quad_multi(W_vals, prod_vals, rem_targets, n_in, rng, trials=18, cur_val=0):
    FULL = (1 << n_in) - 1
    M = len(W_vals)
    P = len(prod_vals)
    d = len(rem_targets)
    assert d >= 1
    base_pivs = {}
    base_c = {}
    def _ins(pivs, piv_c, v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs:
                v ^= pivs[lsb]; comb ^= piv_c[lsb]
            else:
                pivs[lsb] = v; piv_c[lsb] = comb; return
    def _check_hit(pivs, piv_c):
        r0 = cur_val; cs0 = 0
        while r0:
            lsb = r0 & -r0
            if lsb in pivs: r0 ^= pivs[lsb]; cs0 ^= piv_c[lsb]
            else: break
        tp = {}; tcs = {}; tct = {}
        for ti, tv in enumerate(rem_targets):
            v = tv; cs = 0; ct = 1 << ti
            while v:
                lsb = v & -v
                if lsb in pivs: v ^= pivs[lsb]; cs ^= piv_c[lsb]
                elif lsb in tp: v ^= tp[lsb]; cs ^= tcs[lsb]; ct ^= tct[lsb]
                else: tp[lsb] = v; tcs[lsb] = cs; tct[lsb] = ct; break
            else:
                if r0 == 0: return ct, cs0 ^ cs
        if r0 != 0:
            v = r0; cs = cs0; ct = 0
            while v:
                lsb = v & -v
                if lsb in pivs: v ^= pivs[lsb]; cs ^= piv_c[lsb]
                elif lsb in tp: v ^= tp[lsb]; cs ^= tcs[lsb]; ct ^= tct[lsb]
                else: return None
            if ct != 0: return ct, cs
        return None

    _ins(base_pivs, base_c, FULL, 0)
    for i in range(M): _ins(base_pivs, base_c, W_vals[i], 1 + i)
    for p in range(P): _ins(base_pivs, base_c, prod_vals[p], 1 + M + p)

    h0 = _check_hit(base_pivs, base_c)
    if h0 is not None:
        ct, cs = h0
        return ct, cs & 1, (cs >> 1) & ((1 << M) - 1), (cs >> (1 + M)) & ((1 << P) - 1), []

    lin_rk = len(base_pivs)
    max_def = 5  # Pure Python path

    # --- PATCHED: Bilinear oversampling ---
    def _n_u_for_r(r):
        return min(2 * r, M - 1) if M > 2 else r

    r_m = 1
    while r_m < M:
        _nu = _n_u_for_r(r_m)
        if lin_rk + _nu * M - _nu * (_nu + 1) // 2 + d >= n_in - max_def:
            break
        r_m += 1
    r_s = r_m
    while r_s < M:
        _nu = _n_u_for_r(r_s)
        if lin_rk + _nu * M - _nu * (_nu + 1) // 2 >= n_in:
            break
        r_s += 1

    q_off = 1 + M + P
    best = None

    for r_try in range(r_m, r_s + 2):
        n_u = _n_u_for_r(r_try)
        gap = n_in - (lin_rk + n_u * M - n_u * (n_u + 1) // 2 + d)
        n_tr = max(trials * 2, 96) if gap > 0 else max(trials, 48)

        for _ in range(n_tr):
            u_masks = []; u_vecs = []
            for _k in range(n_u):
                um = rng.randrange(1, 1 << M)
                uv = 0
                for a in _iter_bits(um): uv ^= W_vals[a]
                u_masks.append(um); u_vecs.append(uv)
            pivs = dict(base_pivs); piv_c = dict(base_c)
            idx = q_off
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx); idx += 1
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
                    if len(p_uv) <= max(r_try - 1, 1):
                        return best
        if best is not None and len(best[4]) <= r_try:
            return best

    if best is not None:
        return best
    cb, lm, p_uv = solve_low_rank_quad(W_vals, cur_val ^ rem_targets[0], n_in=n_in, rng=rng, try_r_below=1)
    return 1, cb, lm, 0, p_uv


# ===== ORIGINAL _solve_bcast_quad_multi (for comparison) =====

def _solve_bcast_quad_multi_ORIG(W_vals, prod_vals, rem_targets, n_in, rng, trials=18, cur_val=0):
    FULL = (1 << n_in) - 1
    M = len(W_vals); P = len(prod_vals); d = len(rem_targets)
    assert d >= 1
    base_pivs = {}; base_c = {}
    def _ins(pivs, piv_c, v, idx):
        comb = 1 << idx
        while v:
            lsb = v & -v
            if lsb in pivs: v ^= pivs[lsb]; comb ^= piv_c[lsb]
            else: pivs[lsb] = v; piv_c[lsb] = comb; return
    def _check_hit(pivs, piv_c):
        r0 = cur_val; cs0 = 0
        while r0:
            lsb = r0 & -r0
            if lsb in pivs: r0 ^= pivs[lsb]; cs0 ^= piv_c[lsb]
            else: break
        tp = {}; tcs = {}; tct = {}
        for ti, tv in enumerate(rem_targets):
            v = tv; cs = 0; ct = 1 << ti
            while v:
                lsb = v & -v
                if lsb in pivs: v ^= pivs[lsb]; cs ^= piv_c[lsb]
                elif lsb in tp: v ^= tp[lsb]; cs ^= tcs[lsb]; ct ^= tct[lsb]
                else: tp[lsb] = v; tcs[lsb] = cs; tct[lsb] = ct; break
            else:
                if r0 == 0: return ct, cs0 ^ cs
        if r0 != 0:
            v = r0; cs = cs0; ct = 0
            while v:
                lsb = v & -v
                if lsb in pivs: v ^= pivs[lsb]; cs ^= piv_c[lsb]
                elif lsb in tp: v ^= tp[lsb]; cs ^= tcs[lsb]; ct ^= tct[lsb]
                else: return None
            if ct != 0: return ct, cs
        return None

    _ins(base_pivs, base_c, FULL, 0)
    for i in range(M): _ins(base_pivs, base_c, W_vals[i], 1 + i)
    for p in range(P): _ins(base_pivs, base_c, prod_vals[p], 1 + M + p)
    h0 = _check_hit(base_pivs, base_c)
    if h0 is not None:
        ct, cs = h0
        return ct, cs & 1, (cs >> 1) & ((1 << M) - 1), (cs >> (1 + M)) & ((1 << P) - 1), []

    lin_rk = len(base_pivs); max_def = 5
    r_m = 1
    while r_m < M and lin_rk + r_m * M - r_m * (r_m + 1) // 2 + d < n_in - max_def:
        r_m += 1
    r_s = 1
    while r_s < M and lin_rk + r_s * M - r_s * (r_s + 1) // 2 < n_in:
        r_s += 1
    q_off = 1 + M + P; best = None

    for r_try in range(r_m, r_s + 2):
        gap = n_in - (lin_rk + r_try * M - r_try * (r_try + 1) // 2 + d)
        n_tr = max(trials, 56) if gap > 0 else (trials if r_try <= r_s else 12)
        for _ in range(n_tr):
            u_masks = []; u_vecs = []
            for _k in range(r_try):
                um = rng.randrange(1, 1 << M)
                uv = 0
                for a in _iter_bits(um): uv ^= W_vals[a]
                u_masks.append(um); u_vecs.append(uv)
            pivs = dict(base_pivs); piv_c = dict(base_c)
            idx = q_off
            for uv in u_vecs:
                for w in W_vals:
                    _ins(pivs, piv_c, uv & w, idx); idx += 1
            hit = _check_hit(pivs, piv_c)
            if hit is not None:
                ct, cs = hit
                cb0 = cs & 1; lm0 = (cs >> 1) & ((1 << M) - 1)
                bm0 = (cs >> (1 + M)) & ((1 << P) - 1)
                synth_c = cb0 | (lm0 << 1) | ((cs >> q_off) << (1 + M))
                cb, lm, p_uv = _extract_quad_solution(synth_c, M, u_masks)
                if best is None or len(p_uv) < len(best[4]):
                    best = (ct, cb, lm, bm0, p_uv)
                    if len(p_uv) <= r_try - 1: return best
            if best is not None: return best
    if best is not None: return best
    cb, lm, p_uv = solve_low_rank_quad(W_vals, cur_val ^ rem_targets[0], n_in=n_in, rng=rng, try_r_below=1)
    return 1, cb, lm, 0, p_uv


# ===== Test harness =====

def run_one(solver_fn, label, M, k, n_in=256, seed=42):
    rng_data = random.Random(seed)
    W_vals = [rng_data.getrandbits(n_in) for _ in range(M)]
    prods = []
    for i in range(min(M, 5)):
        for j in range(i+1, min(M, 5)):
            prods.append(W_vals[i] & W_vals[j])
    targets = [rng_data.getrandbits(n_in) for _ in range(k)]

    rng = random.Random(seed + 1000)
    total_ccx = 0
    t0 = time.time()
    remaining = list(targets)
    W_live = list(W_vals)
    P_live = list(prods)

    for bit_idx in range(k):
        if not remaining:
            break
        ct, cb, lm, bm, p_uv = solver_fn(W_live, P_live, remaining, n_in, rng, trials=18, cur_val=0)

        # Verify
        goal = 0
        for ti in range(len(remaining)):
            if ct & (1 << ti):
                goal ^= remaining[ti]
        recon = _verify_quad_eval(W_live, p_uv, lm, cb, n_in)
        bcast_val = 0
        for pi in _iter_bits(bm):
            bcast_val ^= P_live[pi]
        recon ^= bcast_val
        assert recon == goal, f"VERIFY FAILED bit {bit_idx}!"

        total_ccx += len(p_uv)
        W_live = list(W_live) + [recon]
        new_rem = []
        for ti in range(len(remaining)):
            if not (ct & (1 << ti)):
                new_rem.append(remaining[ti])
        remaining = new_rem

    elapsed = time.time() - t0
    return total_ccx, elapsed


def main():
    configs = [
        ("5a_c2fwd",   34,  8, 256),
        ("5b_c0r",     30,  4, 256),
        ("9_c2_rev",   23,  8, 256),
        ("10_addr_rev", 26, 8, 256),
    ]
    # Skip EarlyW0 (M=6) and QROM1 (M=8) — too small M for oversampling to help much
    # and they're slow in pure Python with n_in=256

    print("=" * 65)
    print("  GF(2) Solver: ORIGINAL vs PATCHED (bilinear oversampling)")
    print("=" * 65)
    print(f"  {'Stage':<20} {'ORIG CCX':>10} {'PATCH CCX':>10} {'Saved':>8} {'Δtime':>10}")
    print("-" * 65)

    total_orig = 0
    total_patch = 0

    for label, M, k, n_in in configs:
        ccx_o, t_o = run_one(_solve_bcast_quad_multi_ORIG, label, M, k, n_in)
        ccx_p, t_p = run_one(_solve_bcast_quad_multi, label, M, k, n_in)
        saved = ccx_o - ccx_p
        total_orig += ccx_o
        total_patch += ccx_p
        print(f"  {label:<20} {ccx_o:>10} {ccx_p:>10} {saved:>+8} {t_p - t_o:>+10.2f}s")

    print("-" * 65)
    print(f"  {'TOTAL':<20} {total_orig:>10} {total_patch:>10} {total_orig - total_patch:>+8}")
    print()
    print(f"  Note: These are the large-M stages where oversampling helps most.")
    print(f"  EarlyW0/QROM1 (small M) need Numba for speed; tested on Kaggle.")

if __name__ == '__main__':
    main()
