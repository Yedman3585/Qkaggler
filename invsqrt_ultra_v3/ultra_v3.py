"""Ultra-V3 Quantum Circuit for `invsqrt` (Kaggle Google Quantum AI Benchmark I).

Major upgrades over `ultra_v2` (Q=68, C=2,053, score=139,624) targeting Q=58-61, C<1,450, score<86,000:
  1. `PoolAllocator`: eliminates 100% of qubit leaks by reusing all HMR-freed qubits (`addr` + `c0_lo`)
     inside every adder, unpack/pack, streaming shifter, MUX carry, and LUT temp boost.
  2. Deferred `HMR` checkpoints in `pebbled_gidney_iadd`: 0 conditional CCXs in Forward `c2*u` and `r1*u`!
  3. Reverse `HMR(ovf)` with 0-Toffoli Clifford carry-out phase fixup: -1 CCX per reverse row & +1 ancilla!
  4. `v34`-style 2-bit-exact `MUX(u_i, r1[k-1], r1[k-2])` rounding carry-in: cuts `r1*u` rounding error 2.5x,
     enabling `gb=1` (`p_lo_acc=20` bits, huge margin!) and `wc1=17, wp=27, gb=0` (`skip37`)!
  5. Unsplit `c2` (`8` bits) with deferred `HMR(c2_fwd)`, deferred `HMR(c2_ovf)` (reconstructed from
     `c1_orig[w_c2] ^ r1[w_c2] ^ u_c2[top]`), and early `HMR(c0_lo)` before reverse `c2*u` (0 pebbling!).
  6. `uncompute_qrom1_init_and`: greedy `init_and`/`del_and` uncomputation (~16-17 CCX vs 25 CCX `del_lookup`).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import kickmix as km
import numpy as np
from scipy.optimize import linprog

import gf2_lut as gl
import ultra_primitives as up
import ultra_v2_primitives as up2


@dataclass
class UltraV3Config:
    wc1: int = 18
    wc2: int = 8
    d_c2: int = 8
    u_bits: int = 14
    wp: int = 26
    gb: int = 1
    zgb: int = 0               # 1 -> keep bottom `gb` bits of c0_lo constant (saves `gb` QROM bits)
    w_hi: int = 8
    skip37: bool = False       # For wp=27, gb=0: drop 2^-37 rotation at s=0 (saves 1 Z_POW rot!)
    mux_round_r1: bool = True
    mux_r1_min_row: int = 0    # Apply MUX rounding to all r1 rows (mm=+6.1e-12!)
    skip_mux_23: bool = True   # Skip MUX rounding on ONLY rows 2,3 to lock in Q=60 strictly!
    mux_round_c2: bool = False
    use_bcast: bool = True     # Use shared-product CCX broadcasting in QROM1 & all compute_quad_bits!
    n_c0lo_q1: int = 15        # c0_lo bits computed in QROM1 (keeps Stage 2/4/5a/5b/6/7 strictly <= Q=60!)
    n_temp_q1: int = 4         # 1-CCX boost wires during QROM1 (23+18+15+4 = 60!)
    n_boot_q1: int = 11        # 1-CCX bootstrap wires in QROM1_bcast (11+4=15 -> M1=23? auto-clamps >=25)
    n_boost_c0hi: int = 1      # boost wires for c0_hi (15+18+15+7+1=56 <= 60!)
    n_boost_c2fwd: int = 4     # boost wires for unsplit c2_fwd (15+18+15+8+4=60!)
    n_c2_keep_c0r: int = 6     # keep 6 pure c2_q bits alive during c0_lo_rest (48+6+1+5=60!)
    n_boost_c0r: int = 0       # boost wires for c0_lo_rest
    n_c0_drop_c2r: int = 5     # HMR-defer 5 c0_lo bits before c2_rev (keeps >=23 pure wires!)
    n_boost_c2rev: int = 4     # boost wires for c2_rev
    n_boost_addr: int = 18     # boost wires for addr_rev (c0_lo already HMR-freed -> use all pool!)
    lut_seeds: int = 5         # multi-seed best-of-N for each GF(2) LUT (fast dry-run selection!)
    c2_max_anc: int = 2        # forward c2*u ancillas (asymmetric defer_hmr -> 0 cond CCX in fwd & rev!!)
    r1_block_mode: str = '2blk'  # '2blk' (2 ovfs -> strictly Q=60!), '3blk', '4blk'
    r1_anc_budget: int = 7     # total ancillas (ovfs + cin + carries) during r1*u (53+7=60!)
    defer_c2_ovf: bool = True  # HMR-free c2_ovf before r1*u (-1 peak Q!) and recover via c1_orig[w_c2]
    pack_s_in_r1B: bool = True
    early_w0_s3: bool = True   # Run 6-bit linear c0_hi + s3==1 fusion Wave 0 BEFORE QROM1 (-40 Tof-eq!)
    n_tmp_w0: int = 7          # init_and temps for 6-bit Wave 0 LUT (26+12+7+5+6+2 = 58 <= 60!)
    cap_r1_anc_to_pool: bool = True  # Cap r1*u ancillas to pool_avail to avoid bumping Q above 60!
    do_cd: bool = True
    do_lp: bool = True
    seed: int = 7

    @property
    def wp_acc(self) -> int:
        return self.wp + self.gb


def _sdm_np(
    a: np.ndarray, u: np.ndarray, m_bits: int, shift_base: int,
    mux_round: bool = False, mux_min_row: int = 0, skip_23: bool = False,
) -> np.ndarray:
    """Exact NumPy model of signed-digit `sum_i sg_i * round(a * 2^(shift_base + i))`."""
    acc = np.zeros_like(u, dtype=np.int64)
    for i in range(m_bits):
        bit = (u >> i) & 1
        sh = shift_base + i
        if sh >= 0:
            t1 = a << sh
            t0 = t1
        else:
            k = -sh
            base = a >> k
            rk1 = (a >> (k - 1)) & 1
            if mux_round and i >= mux_min_row and not (skip_23 and i in (2, 3)):
                rk2 = ((a >> (k - 2)) & 1) if k >= 2 else 0
                t1 = base + rk1
                t0 = base + rk2
            else:
                t1 = base + rk1
                t0 = t1
        acc += np.where(bit == 1, t1, -t0)
    return acc


def _sdm_group_bounds(
    vals: np.ndarray, rows_desc: list[int], shift_base: int,
    mux_round: bool = False, mux_min_row: int = 0, skip_23: bool = False,
) -> tuple[int, int]:
    """Exact max |sum| and min (term_first - sum_rest) over all values in `vals` for `rows_desc`."""
    max_sum = 0
    min_margin = 10**9
    for v in np.unique(vals).tolist():
        t_maxs = []
        t_mins = []
        for i in rows_desc:
            sh = shift_base + i
            if sh >= 0:
                t_hi = v << sh
                t_lo = t_hi
            else:
                k = -sh
                base = v >> k
                rk1 = (v >> (k - 1)) & 1
                if mux_round and i >= mux_min_row and not (skip_23 and i in (2, 3)):
                    rk2 = ((v >> (k - 2)) & 1) if k >= 2 else 0
                    t_hi = base + max(rk1, rk2)
                    t_lo = base + min(rk1, rk2)
                else:
                    t_hi = base + rk1
                    t_lo = t_hi
            t_maxs.append(t_hi)
            t_mins.append(t_lo)
        s_all = sum(t_maxs)
        if s_all > max_sum:
            max_sum = s_all
        margin = t_mins[0] - sum(t_maxs[1:])
        if margin < min_margin:
            min_margin = margin
    return max_sum, min_margin


def compute_v3_tables(cfg: UltraV3Config) -> dict:
    max_x = 3 * (2**10 - 2) ** 2
    xs = np.arange(1, max_x + 1, dtype=np.int64)
    bl = np.floor(np.log2(xs)).astype(np.int64) + 1
    e = 22 - bl
    x_norm = xs << e
    parity = e & 1
    s_arr = e >> 1
    m_norm = x_norm / float(1 << 21)
    m_eff = np.where(parity == 1, m_norm, 2.0 * m_norm)
    true_m = 1.0 / np.sqrt(m_eff)
    tol_m = (2.0 ** -24) * true_m
    true_x = 1.0 / np.sqrt(xs)
    tol_x = (2.0 ** -24) * true_x

    u_bits = cfg.u_bits
    wp_acc = cfg.wp_acc
    addr = ((1 - parity) << 7) | ((x_norm >> u_bits) & 127)
    u = x_norm & ((1 << u_bits) - 1)

    key = (addr << u_bits) | u
    _, uniq_idx = np.unique(key, return_index=True)
    addr_u, u_u, true_u, tol_u = addr[uniq_idx], u[uniq_idx], true_m[uniq_idx], tol_m[uniq_idx]
    u_c2_u = u_u >> (u_bits - cfg.d_c2)
    # Track for each unique (addr, u) whether s_arr==0 occurs (needed when skip37=True)
    s0_u = (s_arr[uniq_idx] == 0)

    order = np.argsort(addr_u, kind='stable')
    addr_s = addr_u[order]
    starts = np.searchsorted(addr_s, np.arange(256), side='left')
    ends = np.searchsorted(addr_s, np.arange(256), side='right')

    nodes = np.cos((2 * np.arange(3) + 1) * math.pi / 6)
    u_center = 0.5 - 2.0 ** (-(u_bits + 1))
    c0_r, c1_r, c2_r = np.zeros(256), np.zeros(256), np.zeros(256)
    for idx in range(256):
        is24 = idx >> 7
        sub = idx & 127
        hz = (2.0 / 128) if is24 else (1.0 / 128)
        a = (2.0 + sub * hz) if is24 else (1.0 + sub * hz)
        poly = np.polyfit(nodes, 1.0 / np.sqrt(a + hz * u_center + 0.5 * hz * nodes), 2)
        c2_r[idx], c1_r[idx], c0_r[idx] = poly[0], -poly[1], poly[2]

    max_r1_r = float(np.max(c1_r + c2_r))
    c1_shift = math.floor(-math.log2(max_r1_r))
    c2_shift = math.floor(-math.log2(float(np.max(c2_r))))
    while int(round(max_r1_r * (1 << (c1_shift + cfg.wc1)))).bit_length() > cfg.wc1:
        c1_shift -= 1
    while int(round(float(np.max(c2_r)) * (1 << (c2_shift + cfg.wc2)))).bit_length() > cfg.wc2:
        c2_shift -= 1
    c1_scale = c1_shift + cfg.wc1
    c2_scale = c2_shift + cfg.wc2

    c1_tab = np.round(c1_r * (1 << c1_scale)).astype(np.int64)
    c2_tab = np.round(c2_r * (1 << c2_scale)).astype(np.int64)
    c0_tab = np.round(c0_r * (1 << wp_acc)).astype(np.int64)
    c0_step = (1 << cfg.gb) if (cfg.zgb and cfg.gb > 0) else 1
    if cfg.zgb and cfg.gb > 0:
        c0_tab = ((c0_tab >> cfg.gb) << cfg.gb) + (1 << (cfg.gb - 1))

    sh_c2 = c1_scale - c2_scale - cfg.d_c2
    sh_r1 = wp_acc - c1_scale - u_bits

    # Initial centering of c0_tab
    t_c2 = _sdm_np(c2_tab[addr_u], u_c2_u, cfg.d_c2, sh_c2, mux_round=cfg.mux_round_c2)
    r1 = c1_tab[addr_u] - t_c2
    t_r1 = _sdm_np(r1, u_u, u_bits, sh_r1, mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_23=cfg.skip_mux_23)
    p_acc = c0_tab[addr_u] - t_r1
    p_out_u = p_acc >> cfg.gb
    if cfg.skip37:
        p_out_u = np.where(s0_u, p_out_u & ~1, p_out_u)
    err_m = p_out_u / float(1 << cfg.wp) - true_u
    for idx in range(256):
        s0, e0 = starts[idx], ends[idx]
        if s0 == e0:
            continue
        em = err_m[order[s0:e0]]
        delta = int(round(0.5 * (em.max() + em.min()) * (1 << wp_acc)))
        if cfg.zgb and cfg.gb > 0:
            delta = int(round(delta / c0_step)) * c0_step
        c0_tab[idx] -= delta

    p_acc = c0_tab[addr_u] - t_r1
    p_out_u = p_acc >> cfg.gb
    if cfg.skip37:
        p_out_u = np.where(s0_u, p_out_u & ~1, p_out_u)
    rat_arr = np.abs(p_out_u / float(1 << cfg.wp) - true_u) / tol_u

    if cfg.do_cd:
        max_c1, max_c2 = (1 << cfg.wc1) - 1, (1 << cfg.wc2) - 1
        for idx in range(256):
            s0, e0 = starts[idx], ends[idx]
            if s0 == e0:
                continue
            idxs = order[s0:e0]
            if rat_arr[idxs].max() < 0.60:
                continue
            ui, uc2i, tmi, toli, s0i = u_u[idxs], u_c2_u[idxs], true_u[idxs], tol_u[idxs], s0_u[idxs]

            def ev(c0v, c1v, c2v):
                tc2 = _sdm_np(np.full_like(uc2i, c2v), uc2i, cfg.d_c2, sh_c2, mux_round=cfg.mux_round_c2)
                r1v = c1v - tc2
                if r1v.min() < 0 or r1v.max() > max_c1:
                    return 999.0
                tr1 = _sdm_np(r1v, ui, u_bits, sh_r1, mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_23=cfg.skip_mux_23)
                pa = c0v - tr1
                po = pa >> cfg.gb
                if cfg.skip37:
                    po = np.where(s0i, po & ~1, po)
                return float(np.max(np.abs(po / float(1 << cfg.wp) - tmi) / toli))

            b0, b1, b2 = int(c0_tab[idx]), int(c1_tab[idx]), int(c2_tab[idx])
            best = ev(b0, b1, b2)
            for d0 in range(-4 * c0_step, 5 * c0_step, c0_step):
                v = ev(b0 + d0, b1, b2)
                if v < best:
                    best, b0 = v, b0 + d0
            for _ in range(3):
                imp = False
                for step in (2, 1):
                    for d1 in (-step, step):
                        if 0 <= b1 + d1 <= max_c1:
                            for d0 in (-c0_step, 0, c0_step):
                                v = ev(b0 + d0, b1 + d1, b2)
                                if v < best:
                                    best, b0, b1, imp = v, b0 + d0, b1 + d1, True
                    for d2 in (-step, step):
                        if 0 <= b2 + d2 <= max_c2:
                            for d0 in (-c0_step, 0, c0_step):
                                v = ev(b0 + d0, b1, b2 + d2)
                                if v < best:
                                    best, b0, b2, imp = v, b0 + d0, b2 + d2, True
                if not imp:
                    break
            c0_tab[idx], c1_tab[idx], c2_tab[idx] = b0, b1, b2

    u_c2_f = u >> (u_bits - cfg.d_c2)
    t_c2_f = _sdm_np(c2_tab[addr], u_c2_f, cfg.d_c2, sh_c2, mux_round=cfg.mux_round_c2)
    r1_f = c1_tab[addr] - t_c2_f
    assert r1_f.min() >= 0 and r1_f.max() < (1 << cfg.wc1)
    t_r1_f = _sdm_np(r1_f, u, u_bits, sh_r1, mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_23=cfg.skip_mux_23)
    p_acc_f = c0_tab[addr] - t_r1_f

    sh_hi = wp_acc - cfg.w_hi
    c0_hi_tab = np.zeros(256, dtype=np.int64)
    order_f = np.argsort(addr, kind='stable')
    addr_sf = addr[order_f]
    st_f = np.searchsorted(addr_sf, np.arange(256), side='left')
    en_f = np.searchsorted(addr_sf, np.arange(256), side='right')
    for idx in range(256):
        s0, e0 = st_f[idx], en_f[idx]
        c0_hi_tab[idx] = int(p_acc_f[order_f[s0:e0]].min() >> sh_hi) if s0 < e0 else int(c0_tab[idx] >> sh_hi)
    msb_hi = 1 << (cfg.w_hi - 1)
    assert c0_hi_tab.min() >= msb_hi and c0_hi_tab.max() < (1 << cfg.w_hi)
    c0_hi_var = c0_hi_tab - msb_hi

    # Optional exact 6-bit decomposition of c0_hi: c0_hi[4*g+lo] = base6[g] - (f1[g]*(lo>=1) + f2[g]*(lo>=2) + f3[g]*(lo==3))
    # 100% EXACT (0 change to c0_hi / c0_lo / plo_acc!) on all 256 addresses!
    base6_var_list: list[int] | None = None
    f123_list: list[int] | None = None
    top3_64_list: list[int] | None = None
    s3_cut = 16
    if cfg.early_w0_s3:
        base6_var_list = [int(c0_hi_tab[4 * g] - msb_hi) for g in range(64)]
        f123_list = []
        for g in range(64):
            d1 = int(c0_hi_tab[4 * g] - c0_hi_tab[4 * g + 1])
            d2 = int(c0_hi_tab[4 * g + 1] - c0_hi_tab[4 * g + 2])
            d3 = int(c0_hi_tab[4 * g + 2] - c0_hi_tab[4 * g + 3])
            assert d1 in (0, 1) and d2 in (0, 1) and d3 in (0, 1), f'Non-unit step at group {g}: {(d1,d2,d3)}'
            f123_list.append(d1 | (d2 << 1) | (d3 << 2))

    c0_lo_acc_tab = c0_tab - (c0_hi_tab << sh_hi)
    p_lo_acc_f = p_acc_f - (c0_hi_tab[addr] << sh_hi)
    assert p_lo_acc_f.min() >= 0 and c0_lo_acc_tab.min() >= 0
    plo_acc_bits = int(max(int(p_lo_acc_f.max()), int(c0_lo_acc_tab.max()))).bit_length()
    plo_bits = plo_acc_bits - cfg.gb

    if cfg.zgb and cfg.gb > 0:
        gb_const = 1 << (cfg.gb - 1)
        assert np.all((c0_lo_acc_tab & ((1 << cfg.gb) - 1)) == gb_const)
        c0_lo_qrom_tab = c0_lo_acc_tab >> cfg.gb
        c0_lo_qrom_bits = plo_acc_bits - cfg.gb
    else:
        gb_const = 0
        c0_lo_qrom_tab = c0_lo_acc_tab
        c0_lo_qrom_bits = plo_acc_bits

    # Check injectivity of c0_lo_qrom_tab (needed if defer_c2_ovf=True!)
    c0lo_unique = len(np.unique(c0_lo_qrom_tab))

    p_lo_f = p_lo_acc_f >> cfg.gb
    assert np.array_equal((c0_hi_tab[addr] << (cfg.wp - cfg.w_hi)) + p_lo_f, p_acc_f >> cfg.gb)

    if cfg.early_w0_s3:
        # Fuse p_lo[16..18] for s>=8 (where u==0 and addr&3==0 ALWAYS!) into Wave 0!
        # CRITICAL: u==0 in signed-digit arithmetic has NON-ZERO r1*u_signed! Compute exact p_lo at u==0!
        z256 = np.zeros(256, dtype=np.int64)
        c2_u0 = _sdm_np(c2_tab, z256, cfg.d_c2, sh_c2, mux_round=cfg.mux_round_c2)
        r1_u0 = c1_tab - c2_u0
        p_lo_u0 = (c0_lo_acc_tab - _sdm_np(
            r1_u0, z256, cfg.u_bits, sh_r1, mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_23=cfg.skip_mux_23,
        )) >> cfg.gb
        # Guard against (255 << 2) + 4 == 1024 overflow at addr=0 (M=1.0) when skip_mux_23 nudges p_lo_u0[0] to 2^18:
        if c0_hi_tab[0] == 2 * msb_hi - 1 and int(p_lo_u0[0] >> s3_cut) >= 4:
            delta_nudge = (int(p_lo_u0[0]) - ((4 << s3_cut) - 1)) << cfg.gb
            c0_tab[0] -= delta_nudge
            c0_lo_acc_tab[0] -= delta_nudge
            p_acc_f = np.where(addr == 0, p_acc_f - delta_nudge, p_acc_f)
            p_lo_acc_f = np.where(addr == 0, p_lo_acc_f - delta_nudge, p_lo_acc_f)
            p_lo_f = p_lo_acc_f >> cfg.gb
            p_lo_u0[0] -= (delta_nudge >> cfg.gb)
            if cfg.zgb and cfg.gb > 0:
                c0_lo_qrom_tab = c0_lo_acc_tab >> cfg.gb
            else:
                c0_lo_qrom_tab = c0_lo_acc_tab
        top3_64_arr = (p_lo_u0[::4] >> s3_cut) & 7
        assert np.all((c0_hi_tab[::4] << 2) + top3_64_arr < (1 << (cfg.w_hi + 2))), 'c0_hi_s3 overflow!'
        top3_64_list = top3_64_arr.tolist()
        is_s3 = s_arr >= 8
        y_hi_f = (c0_hi_tab[addr] << s_arr) + np.where(is_s3, (p_lo_f >> s3_cut) << np.maximum(0, s_arr - 2), 0)
        y_lo_f = np.where(is_s3, (p_lo_f & ((1 << s3_cut) - 1)) << s_arr, p_lo_f << s_arr)
    else:
        y_hi_f = c0_hi_tab[addr] << s_arr
        y_lo_f = p_lo_f << s_arr
    if cfg.skip37:
        # Drop bit 0 of y_lo_f (which only appears at s_arr==0, bit 0 = 2^-37!)
        y_lo_eff = y_lo_f & ~1
    else:
        y_lo_eff = y_lo_f
    max_lo_bit = int(y_lo_f.max()).bit_length()

    f_hi = cfg.w_hi + 10
    f_lo = cfg.wp + 10
    nom_turns = y_hi_f / float(1 << f_hi) + y_lo_eff / float(1 << f_lo)
    err_x_f = np.abs(nom_turns - true_x)
    rat = float(np.max(err_x_f / tol_x))
    mm = float(np.min(tol_x - err_x_f))

    # Exact unsplit c2 bounds
    c2_rows = list(range(cfg.d_c2 - 1, -1, -1))
    max_c2_s, min_c2_m = _sdm_group_bounds(c2_tab, c2_rows, sh_c2, mux_round=cfg.mux_round_c2)
    assert min_c2_m >= 0, f'c2 1-ovf margin negative: {min_c2_m}'
    w_c2 = max(1, max_c2_s.bit_length())
    assert w_c2 < cfg.wc1, f'w_c2={w_c2} >= wc1={cfg.wc1}'

    # Exact r1 * u block bounds
    r1_unique = np.unique(r1_f)
    if cfg.r1_block_mode == '2blk':
        r1_grps = [[13, 12, 11, 10], [9, 8, 7, 6], [5, 4, 3, 2, 1, 0]]
    elif cfg.r1_block_mode == '4blk':
        r1_grps = [[13, 12, 11], [10, 9, 8], [7, 6, 5], [4, 3, 2], [1, 0]]
    else:  # '3blk'
        r1_grps = [[13, 12, 11], [10, 9, 8], [7, 6, 5], [4, 3, 2, 1, 0]]
    r1_block_widths: list[int | None] = [None]
    for grp in r1_grps[1:]:
        ms, mm_g = _sdm_group_bounds(r1_unique, grp, sh_r1, mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_23=cfg.skip_mux_23)
        assert mm_g >= 0, f'r1 block {grp} 1-ovf margin negative: {mm_g}'
        wb = max(1, ms.bit_length())
        r1_block_widths.append(wb if wb + 2 < plo_acc_bits else None)

    # Nominal vs LP Z_POW angles (half-turns Fractions!)
    angs_hi = [Fraction(2) ** (k - f_hi + 1) if k < f_hi else Fraction(0) for k in range(cfg.w_hi + 10)]
    angs_lo = [
        Fraction(0) if (cfg.skip37 and k == 0) else Fraction(2) ** (k - f_lo + 1)
        for k in range(max_lo_bit)
    ]

    if cfg.do_lp:
        n_hi_rot = f_hi - 3
        lo_k_start = 1 if cfg.skip37 else 0
        lo_ks = list(range(lo_k_start, max_lo_bit))
        n_lo_rot = len(lo_ks)
        n_rot = n_hi_rot + n_lo_rot
        res0 = nom_turns - true_x
        margin0 = tol_x - np.abs(res0)
        active = set(np.argsort(margin0)[:3000].tolist())
        active.update(np.argsort(-np.abs(res0) / tol_x)[:1000].tolist())
        scale = 1e12
        best_delta = np.zeros(n_rot, dtype=np.float64)
        best_mm = mm
        for _it in range(6):
            act_idx = np.array(sorted(active), dtype=np.int64)
            yh_a = y_hi_f[act_idx]
            yl_a = y_lo_f[act_idx]
            B = np.empty((len(act_idx), n_rot), dtype=np.float64)
            for k in range(n_hi_rot):
                B[:, k] = ((yh_a >> k) & 1).astype(np.float64)
            for j, k in enumerate(lo_ks):
                B[:, n_hi_rot + j] = ((yl_a >> k) & 1).astype(np.float64)
            r_a = res0[act_idx] * scale
            tol_a = tol_x[act_idx] * scale
            m_act = len(act_idx)
            A_ub = np.vstack([np.hstack([B, np.ones((m_act, 1))]), np.hstack([-B, np.ones((m_act, 1))])])
            b_ub = np.concatenate([tol_a - r_a, tol_a + r_a])
            c_obj = np.zeros(n_rot + 1)
            c_obj[-1] = -1.0
            sol = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(-1e4, 1e4)] * n_rot + [(None, None)], method='highs')
            if not sol.success:
                break
            delta = sol.x[:n_rot] / scale
            adj = np.zeros(len(xs), dtype=np.float64)
            for k in range(n_hi_rot):
                if delta[k] != 0:
                    adj += delta[k] * ((y_hi_f >> k) & 1)
            for j, k in enumerate(lo_ks):
                if delta[n_hi_rot + j] != 0:
                    adj += delta[n_hi_rot + j] * ((y_lo_f >> k) & 1)
            err_new = np.abs(res0 + adj)
            margin_new = tol_x - err_new
            mm_new = float(margin_new.min())
            if mm_new > best_mm:
                best_mm = mm_new
                best_delta = delta.copy()
                rat = float(np.max(err_new / tol_x))
            before = len(active)
            active.update(np.argsort(margin_new)[:1000].tolist())
            if mm_new >= sol.x[-1] / scale - 1e-15 or len(active) == before:
                break
        if best_mm > mm:
            mm = best_mm
            for k in range(n_hi_rot):
                if best_delta[k] != 0:
                    angs_hi[k] = Fraction(float(angs_hi[k]) + 2.0 * float(best_delta[k]))
            for j, k in enumerate(lo_ks):
                if best_delta[n_hi_rot + j] != 0:
                    angs_lo[k] = Fraction(float(angs_lo[k]) + 2.0 * float(best_delta[n_hi_rot + j]))

    return {
        'c0_hi_var': c0_hi_var.tolist(),
        'base6_var': base6_var_list,
        'f123': f123_list,
        'top3_64': top3_64_list,
        's3_cut': s3_cut,
        'c0_lo_qrom_tab': c0_lo_qrom_tab.tolist(),
        'c1_tab': c1_tab.tolist(),
        'c2_tab': c2_tab.tolist(),
        'sh_c2': sh_c2,
        'sh_r1': sh_r1,
        'w_c2': w_c2,
        'c2_rows': c2_rows,
        'r1_grps': r1_grps,
        'r1_block_widths': r1_block_widths,
        'plo_acc_bits': plo_acc_bits,
        'plo_bits': plo_bits,
        'c0_lo_qrom_bits': c0_lo_qrom_bits,
        'c0lo_unique': c0lo_unique,
        'gb_const': gb_const,
        'max_lo_bit': max_lo_bit,
        'angs_hi': angs_hi,
        'angs_lo': angs_lo,
        'rat': rat,
        'mm': mm,
    }


def _apply_c2_unsplit_1ovf(
    builder: km.CircuitBuilder,
    uncompute: bool,
    u_c2: Sequence[km.q],
    c2_q: Sequence[km.q],
    c1_lo: Sequence[km.q],
    ovf: km.q | None,
    rows: list[int],
    shift_base: int,
    max_anc: int,
    alloc: up.PoolAllocator,
    defer_tokens: dict,
    phase_cout_bit: km.b | None = None,
    mux_round: bool = False,
) -> None:
    """Unsplit `c2 * u_c2` (1-ovf block). In reverse, if `phase_cout_bit is not None`, `ovf` has
    already been HMR-freed so reverse runs on `c1_lo` (1 bit shorter, -1 CCX per row!)."""
    inv_sg = False  # subtract=True
    lo_tgt = list(c1_lo)
    i0 = rows[0]

    def _tok(i: int):
        return defer_tokens.setdefault(('c2', i), {})

    if not uncompute:
        assert ovf is not None
        ext_tgt = lo_tgt + [ovf]
        up2._signed_digit_iadd_row(
            builder, False, u_c2[i0], inv_sg, c2_q, ext_tgt, shift_base + i0,
            max_anc=max_anc, flip_target_before=lo_tgt, flip_target_after=ext_tgt,
            alloc=alloc, defer_hmr=_tok(i0), mux_round=mux_round,
        )
        for i in rows[1:]:
            up2._signed_digit_iadd_row(
                builder, False, u_c2[i], inv_sg, c2_q, ext_tgt, shift_base + i,
                max_anc=max_anc, alloc=alloc, defer_hmr=_tok(i), mux_round=mux_round,
            )
    else:
        if phase_cout_bit is not None:
            # Phase (-1)^(b_ovf * neg_0) where neg_0 = u_c2[i0] (inv_sg=False)
            builder.push_condition(phase_cout_bit)
            builder.z(u_c2[i0])
            builder.pop_condition()
            for i in reversed(rows[1:]):
                up2._signed_digit_iadd_row(
                    builder, True, u_c2[i], inv_sg, c2_q, lo_tgt, shift_base + i,
                    max_anc=max_anc, alloc=alloc, defer_hmr=_tok(i),
                    phase_cout_bit=phase_cout_bit, mux_round=mux_round,
                )
            up2._signed_digit_iadd_row(
                builder, True, u_c2[i0], inv_sg, c2_q, lo_tgt, shift_base + i0,
                max_anc=max_anc, alloc=alloc, defer_hmr=_tok(i0),
                phase_cout_bit=phase_cout_bit, mux_round=mux_round,
            )
        else:
            assert ovf is not None
            ext_tgt = lo_tgt + [ovf]
            for i in reversed(rows[1:]):
                up2._signed_digit_iadd_row(
                    builder, True, u_c2[i], inv_sg, c2_q, ext_tgt, shift_base + i,
                    max_anc=max_anc, alloc=alloc, defer_hmr=_tok(i), mux_round=mux_round,
                )
            up2._signed_digit_iadd_row(
                builder, True, u_c2[i0], inv_sg, c2_q, ext_tgt, shift_base + i0,
                max_anc=max_anc, flip_target_before=lo_tgt, flip_target_after=ext_tgt,
                alloc=alloc, defer_hmr=_tok(i0), mux_round=mux_round,
            )


def _apply_r1_packed_rows(
    builder: km.CircuitBuilder,
    uncompute: bool,
    x: km.array,
    nz: km.q,
    c1: Sequence[km.q],
    p_lo_acc: Sequence[km.q],
    ovf: km.q,
    grp: list[int],
    w_blk: int,
    shift_base: int,
    max_anc: int,
    alloc: up.PoolAllocator,
    defer_tokens: dict,
    hmr_ovf: bool = True,
    mux_round: bool = True,
    mux_min_row: int = 4,
    skip_mux_23: bool = False,
    blk_key: str = 'last',
) -> None:
    """Applies (or undoes with HMR(ovf)!) a 1-ovf block `grp` decoding packed `u[3..0]` on-the-fly!"""
    inv_sg = False  # subtract=True
    lo_tgt = list(p_lo_acc[:w_blk])
    ext_tgt = lo_tgt + [ovf]
    hi_tgt = list(p_lo_acc[w_blk:])
    i0 = grp[0]
    assert i0 >= 4 or (not uncompute and i0 >= 0)

    def _tok(tag):
        return defer_tokens.setdefault((blk_key, tag), {})

    def _run_row(
        i: int, rev: bool, is_first: bool, ge2_q: km.q | None, u_tmp: km.q | None,
        use_lo_only: bool = False, phase_bit: km.b | None = None, row_ma: int = max_anc,
    ):
        tgt = lo_tgt if use_lo_only else ext_tgt
        f_pre = lo_tgt if (is_first or use_lo_only) else ext_tgt
        f_post = lo_tgt if use_lo_only else ext_tgt
        use_mux = mux_round and (i >= mux_min_row) and not (skip_mux_23 and i in (2, 3))
        # Never exceed live pool_avail (keeps Q <= 60 strictly!), while keeping eff_anc >= 4 so n=10 fits 1-level staircase!
        if alloc is not None:
            min_ma = 5 if use_mux else 4
            row_ma = min(row_ma, max(min_ma, len(alloc.avail)))
        if i >= 4:
            up2._signed_digit_iadd_row(
                builder, rev, x[i], inv_sg, c1, tgt, shift_base + i,
                max_anc=row_ma, flip_target_before=f_pre, flip_target_after=f_post,
                alloc=alloc, defer_hmr=_tok(i), phase_cout_bit=phase_bit, mux_round=use_mux,
            )
        elif i in (2, 3):
            assert ge2_q is not None and u_tmp is not None
            builder.x(ge2_q)
            builder.init_and(x[i], ge2_q, u_tmp)
            builder.x(ge2_q)
            up2._signed_digit_iadd_row(
                builder, rev, u_tmp, inv_sg, c1, tgt, shift_base + i,
                max_anc=row_ma, flip_target_before=f_pre, flip_target_after=f_post,
                alloc=alloc, defer_hmr=_tok(i), phase_cout_bit=phase_bit, mux_round=use_mux,
            )
            builder.x(ge2_q)
            builder.del_and(x[i], ge2_q, u_tmp)
            builder.x(ge2_q)
        else:
            assert u_tmp is not None
            builder.x(nz)
            builder.init_and(x[i], nz, u_tmp)
            builder.x(nz)
            up2._signed_digit_iadd_row(
                builder, rev, u_tmp, inv_sg, c1, tgt, shift_base + i,
                max_anc=row_ma, flip_target_before=f_pre, flip_target_after=f_post,
                alloc=alloc, defer_hmr=_tok(i), phase_cout_bit=phase_bit, mux_round=use_mux,
            )
            builder.x(nz)
            builder.del_and(x[i], nz, u_tmp)
            builder.x(nz)

    def _compute_ge2(ge2_q: km.q, or01_q: km.q, rev: bool):
        if not rev:
            builder.x(x[0]); builder.x(x[1])
            builder.init_and(x[0], x[1], or01_q)
            builder.x(x[0]); builder.x(x[1]); builder.x(or01_q)
            builder.init_and(nz, or01_q, ge2_q)
            builder.x(or01_q); builder.x(x[0]); builder.x(x[1])
            builder.del_and(x[0], x[1], or01_q)
            builder.x(x[0]); builder.x(x[1])
        else:
            builder.x(x[0]); builder.x(x[1])
            builder.init_and(x[0], x[1], or01_q)
            builder.x(x[0]); builder.x(x[1]); builder.x(or01_q)
            builder.del_and(nz, or01_q, ge2_q)
            builder.x(or01_q); builder.x(x[0]); builder.x(x[1])
            builder.del_and(x[0], x[1], or01_q)
            builder.x(x[0]); builder.x(x[1])

    rows_ge4 = [i for i in grp if i >= 4]
    rows_23 = [i for i in grp if i in (2, 3)]
    rows_01 = [i for i in grp if i in (0, 1)]
    assert i0 >= 4, 'Packed block top row must have i0 >= 4!'

    if not uncompute:
        for idx_r, i in enumerate(rows_ge4):
            _run_row(i, False, is_first=(idx_r == 0), ge2_q=None, u_tmp=None, row_ma=max_anc)
        if rows_23:
            ge2_q, or01_q = alloc.alloc(2)
            _compute_ge2(ge2_q, or01_q, rev=False)
            for i in rows_23:
                # Forward uses asymmetric deferred-HMR chain (even eff_anc=2 has 0 cond CCX!);
                # budget max(3, max_anc - 2) -> if mux_round=True, eff_anc=2 -> strictly <= Q=60!
                _run_row(i, False, is_first=False, ge2_q=ge2_q, u_tmp=or01_q, row_ma=max(3, max_anc - 2))
            _compute_ge2(ge2_q, or01_q, rev=True)
            alloc.free([or01_q, ge2_q])
        if rows_01:
            u_tmp = alloc.alloc(1)[0]
            for i in rows_01:
                _run_row(i, False, is_first=False, ge2_q=None, u_tmp=u_tmp, row_ma=max(3, max_anc - 1))
            alloc.free(u_tmp)
        up2.propagate_1ovf_carry(
            builder, False, x[i0], True, ovf, hi_tgt, max_anc=max_anc, alloc=alloc, defer_hmr=_tok('prop')
        )
    else:
        up2.propagate_1ovf_carry(
            builder, True, x[i0], True, ovf, hi_tgt, max_anc=max_anc, alloc=alloc, defer_hmr=_tok('prop')
        )
        b_ovf: km.b | None = None
        if hmr_ovf:
            b_ovf = builder.alloc_bits(1)[0]
            builder.hmr(ovf, b_ovf)
            # Immediately return `ovf` to pool (+1 free ancilla during ALL reverse rows!)
            alloc.free(ovf)
            builder.push_condition(b_ovf)
            builder.z(x[i0])
            builder.pop_condition()
        rev_bonus = 1 if hmr_ovf else 0
        if rows_01:
            u_tmp = alloc.alloc(1)[0]
            for i in reversed(rows_01):
                _run_row(
                    i, True, is_first=False, ge2_q=None, u_tmp=u_tmp,
                    use_lo_only=hmr_ovf, phase_bit=b_ovf, row_ma=max(4, max_anc + rev_bonus - 1),
                )
            alloc.free(u_tmp)
        if rows_23:
            ge2_q, or01_q = alloc.alloc(2)
            _compute_ge2(ge2_q, or01_q, rev=False)
            for i in reversed(rows_23):
                _run_row(
                    i, True, is_first=False, ge2_q=ge2_q, u_tmp=or01_q,
                    use_lo_only=hmr_ovf, phase_bit=b_ovf, row_ma=max(4, max_anc + rev_bonus - 2),
                )
            _compute_ge2(ge2_q, or01_q, rev=True)
            alloc.free([or01_q, ge2_q])
        for idx_r in range(len(rows_ge4) - 1, -1, -1):
            _run_row(
                rows_ge4[idx_r], True, is_first=(idx_r == 0), ge2_q=None, u_tmp=None,
                use_lo_only=hmr_ovf, phase_bit=b_ovf, row_ma=max_anc + rev_bonus,
            )
        if b_ovf is not None:
            builder.free(b_ovf)


def _pick_best_lut_seed(
    fn, n_w: int, w_ms: list[int], n_out: int, out_ms: list[int], n_tmp: int,
    base_seed: int, n_seeds: int, **kwargs,
) -> int:
    """Dry-runs `fn` across `n_seeds` seeds on scratch builders and returns the lowest-CCX seed."""
    if n_seeds <= 1:
        return base_seed
    best_s = base_seed
    best_c = 1 << 30
    for k in range(n_seeds):
        s = base_seed + k * 101
        tb = km.CircuitBuilder()
        wq = list(tb.alloc_qubits(n_w))
        oq = list(tb.alloc_qubits(n_out))
        tq = list(tb.alloc_qubits(n_tmp)) if n_tmp > 0 else []
        try:
            c = fn(tb, wq, w_ms, oq, out_ms, temp_qubits=tq, seed=s, **kwargs)
        except Exception:
            continue
        if c < best_c:
            best_c = c
            best_s = s
    return best_s


def _pick_best_qrom1_seed(
    fn, n_addr: int, n_out: int, out_ms: list[int], n_tmp: int,
    base_seed: int, n_seeds: int, **kwargs,
) -> int:
    """Dry-runs QROM1 `fn` across `n_seeds` seeds on scratch builders and returns the lowest-CCX seed."""
    if n_seeds <= 1:
        return base_seed
    best_s = base_seed
    best_c = 1 << 30
    for k in range(n_seeds):
        s = base_seed + k * 101
        tb = km.CircuitBuilder()
        aq = list(tb.alloc_qubits(n_addr))
        oq = list(tb.alloc_qubits(n_out))
        tq = list(tb.alloc_qubits(n_tmp)) if n_tmp > 0 else []
        try:
            c = fn(tb, aq, oq, out_ms, temp_qubits=tq, seed=s, **kwargs)
        except Exception:
            continue
        if c < best_c:
            best_c = c
            best_s = s
    return best_s


def _apply_6bit_linear_c0_hi_wave0(
    b: km.CircuitBuilder,
    x: km.array,
    s_arr: list[km.q],
    info: dict,
    chi_var_bits: int,
    n_tmp: int = 6,
    seed: int = 8,
    n_seeds: int = 12,
) -> int:
    """Runs Wave 0 BEFORE QROM1 (using unpacked s_arr, 0 unpack_s/pack_s!) with:
      - 6-bit GF(2) LUT on addr[2..7] for (base6_var[7], sl6[2], top3[3]) (31-32 CCX, 0-CCX uncomp!)
      - 8-CCX init_and/del_and computation of d = pats[sl6][addr[0..1]], d_nz, s3_b18 (0-CCX rev!)
      - 6-CCX 7-bit Gidney adder base += (s3_b18 - d) keeping carries alive during Wave 0 (0-CCX rev!)
      - Wave 0 streaming phase shift with s3_extra_lo=[b16, b17] (0 extra CCX in Wave 0!)
    """
    a6_q = list(x[16:22])
    a6_m = gl.addr_bit_masks(6)
    lo0, lo1 = x[14], x[15]

    base_m = gl.int_tables_to_bit_masks(info['base6_var'], chi_var_bits)
    f_m = gl.int_tables_to_bit_masks(info['f123'], 3)
    top3_m = gl.int_tables_to_bit_masks(info['top3_64'], 3)
    all_m = base_m + f_m + top3_m

    base_q = list(b.alloc_qubits(chi_var_bits))
    f_q = list(b.alloc_qubits(3))
    top3_q = list(b.alloc_qubits(3))
    out_q = base_q + f_q + top3_q
    tmp_q = list(b.alloc_qubits(n_tmp))
    keep_out: dict = {}
    s_w0 = _pick_best_lut_seed(
        gl.compute_quad_bits_bcast, 6, a6_m, len(out_q), all_m, n_tmp, seed, n_seeds,
        n_in=64, use_init_and_temp=True,
    )
    ccx_lut = gl.compute_quad_bits_bcast(
        b, a6_q, a6_m, out_q, all_m, temp_qubits=tmp_q, n_in=64, seed=s_w0,
        use_init_and_temp=True, keep_temp_out=keep_out,
    )

    f1, f2, f3 = f_q[0], f_q[1], f_q[2]
    b16, b17, b18 = top3_q[0], top3_q[1], top3_q[2]

    # Exact 7-init_and computation of d = f1*(lo>=1) + f2*(lo>=2) + f3*(lo==3), d_nz, s3_b18!
    # Free m3 before Wave 0 (0-CCX del_and) and recompute after (+1 CCX) to keep peak Q <= 60!
    anc_d = list(b.alloc_qubits(6))
    qa, qb, qc, d1, d_nz, s3_b18 = anc_d
    m3 = b.alloc_qubits(1)[0]

    b.init_and(lo0, lo1, m3)  # m3 = (lo == 3)
    # m1 = (lo >= 1) = lo0 ^ lo1 ^ m3 (computed in-place on lo0, restored immediately!)
    b.cx(lo1, lo0)
    b.cx(m3, lo0)
    b.init_and(f1, lo0, qa)
    b.cx(m3, lo0)
    b.cx(lo1, lo0)
    b.init_and(f2, lo1, qb)
    b.init_and(f3, m3, qc)
    b.del_and(lo0, lo1, m3)
    b.free(m3)

    # Full adder (qa, qb, qc) -> d1 = maj(qa, qb, qc), d0 = qa^qb^qc (stored on qa!)
    b.cx(qc, qa)
    b.cx(qc, qb)
    b.init_and(qa, qb, d1)
    b.cx(qc, d1)
    b.cx(qb, qa)  # qa was qa_orig^qc, qb is qb_orig^qc -> qa becomes qa_orig^qb_orig
    b.cx(qc, qa)  # qa becomes qa_orig^qb_orig^qc = d0!
    b.cx(qc, qb)  # restore qb
    d0 = qa

    # d_nz = d0 | d1 = ~(~d0 & ~d1)
    b.x(d0)
    b.x(d1)
    b.init_and(d0, d1, d_nz)
    b.x(d0)
    b.x(d1)
    b.x(d_nz)

    # s3_b18 = s[3] & b18
    b.init_and(s_arr[3], b18, s3_b18)

    # Convert (d0, d1, s3_b18) -> (delta0 = d0 ^ s3_b18 on d0, delta1 = d0_orig ^ d1 on d1)!
    b.cx(d0, d1)
    b.cx(s3_b18, d0)
    delta0, delta1 = d0, d1

    # 6-CCX 7-bit Gidney adder: base[0..6] += [delta0, delta1, d_nz, d_nz, d_nz, d_nz, d_nz]
    carries = list(b.alloc_qubits(chi_var_bits - 1))
    b.init_and(base_q[0], delta0, carries[0])
    b.cx(delta0, base_q[0])
    for i in range(1, chi_var_bits - 1):
        di = delta1 if i == 1 else d_nz
        b.cx(carries[i - 1], di)
        b.cx(carries[i - 1], base_q[i])
        b.init_and(di, base_q[i], carries[i])
        b.cx(carries[i - 1], carries[i])
        b.cx(carries[i - 1], di)
        b.cx(di, base_q[i])
    b.cx(carries[-1], base_q[-1])
    b.cx(d_nz, base_q[-1])

    # Wave 0 streaming phase shift (reusing unpacked x[0], x[2], x[3] and s_arr!)
    up2.streaming_phase_shift_reuse(
        b, base_q, s_arr, [x[0], x[2], x[3]], info['angs_hi'],
        max_s=10, implicit_msb=True, s3_extra_lo=[b16, b17],
    )

    # 0-CCX reverse of 7-bit Gidney adder!
    b.cx(d_nz, base_q[-1])
    b.cx(carries[-1], base_q[-1])
    for i in range(chi_var_bits - 2, 0, -1):
        di = delta1 if i == 1 else d_nz
        b.cx(di, base_q[i])
        b.cx(carries[i - 1], di)
        b.cx(carries[i - 1], carries[i])
        b.del_and(di, base_q[i], carries[i])
        b.cx(carries[i - 1], di)
        b.cx(carries[i - 1], base_q[i])
    b.cx(delta0, base_q[0])
    b.del_and(base_q[0], delta0, carries[0])
    b.free(km.array(carries))

    # 0-CCX reverse of delta0, delta1, s3_b18, d_nz, d1, qa, qb, qc (+1 CCX for transient m3)!
    b.cx(s3_b18, d0)
    b.cx(d0, d1)
    b.del_and(s_arr[3], b18, s3_b18)
    b.x(d_nz)
    b.x(d0)
    b.x(d1)
    b.del_and(d0, d1, d_nz)
    b.x(d0)
    b.x(d1)
    # Undo full adder (qa=d0 -> qa_orig, d1 -> |0>)
    b.cx(qc, qb)
    b.cx(qc, qa)
    b.cx(qb, qa)
    b.cx(qc, d1)
    b.del_and(qa, qb, d1)
    b.cx(qc, qb)
    b.cx(qc, qa)
    # Recompute transient m3 to del_and qc and qa!
    m3 = b.alloc_qubits(1)[0]
    b.init_and(lo0, lo1, m3)
    b.del_and(f3, m3, qc)
    b.del_and(f2, lo1, qb)
    b.cx(lo1, lo0)
    b.cx(m3, lo0)
    b.del_and(f1, lo0, qa)
    b.cx(m3, lo0)
    b.cx(lo1, lo0)
    b.del_and(lo0, lo1, m3)
    b.free(m3)
    b.free(km.array(anc_d))

    # 0-CCX HMR+CZ uncompute of (base_q, f_q, top3_q) using kept (a6_q + tmp_q) (+ tiny transient init_and top-up if <64 quad rank; Q=46 here!)
    cur_q_u = list(keep_out['cur_q'])
    cur_v_u = list(keep_out['cur_v'])
    extra_recs: list[tuple[km.q, int, int, int, int, int]] = []
    extra_qs: list[km.q] = []
    # Check quadratic span of cur_v_u on n_in=64; top up with 1-CCX init_ands if < 64!
    def _quad_rank64(vs: list[int]) -> int:
        pivs: dict[int, int] = {}
        def _add(v: int) -> None:
            while v:
                lsb = v & -v
                if lsb in pivs:
                    v ^= pivs[lsb]
                else:
                    pivs[lsb] = v
                    return
        _add((1 << 64) - 1)
        for v in vs:
            _add(v)
        for i in range(len(vs)):
            vi = vs[i]
            for j in range(i + 1, len(vs)):
                _add(vi & vs[j])
        return len(pivs)

    import random as _rand
    _rng_u = _rand.Random(seed + 999)
    FULL64 = (1 << 64) - 1
    while _quad_rank64(cur_v_u) < 64 and len(extra_qs) < 6:
        tq = b.alloc_qubits(1)[0]
        M_c = len(cur_q_u)
        best_c = (-1, 0, 0, 0, 0, 0)
        for _ in range(40):
            um = _rng_u.randrange(1, 1 << M_c)
            vm = _rng_u.randrange(1, 1 << M_c)
            if um == vm:
                continue
            uc, vc = _rng_u.randrange(2), _rng_u.randrange(2)
            lv = FULL64 if uc else 0
            for a in gl._iter_bits(um):
                lv ^= cur_v_u[a]
            rv = FULL64 if vc else 0
            for bi in gl._iter_bits(vm):
                rv ^= cur_v_u[bi]
            pr = lv & rv
            rk = _quad_rank64(cur_v_u + [pr])
            if rk > best_c[0]:
                best_c = (rk, um, vm, uc, vc, pr)
                if rk == 64:
                    break
        _, um, vm, uc, vc, pr = best_c
        gl._apply_init_and_linear(b, cur_q_u, tq, um, vm, uc, vc, uncompute=False)
        extra_recs.append((tq, M_c, um, vm, uc, vc))
        extra_qs.append(tq)
        cur_q_u.append(tq)
        cur_v_u.append(pr)

    gl.uncompute_bits_hmr(b, cur_q_u, cur_v_u, out_q, all_m, n_in=64)
    for tq, M_prev, um, vm, uc, vc in reversed(extra_recs):
        gl._apply_init_and_linear(b, cur_q_u[:M_prev], tq, um, vm, uc, vc, uncompute=True)
    if extra_qs:
        b.free(km.array(extra_qs))
    for tq, M_prev, um, vm, uc, vc in reversed(keep_out['recs']):
        gl._apply_init_and_linear(b, keep_out['cur_q'][:M_prev], tq, um, vm, uc, vc, uncompute=True)
    b.free(km.array(tmp_q))
    b.free(km.array(out_q))
    return ccx_lut + 8 + (chi_var_bits - 1) + len(extra_qs)


def build_ultra_v3(cfg: UltraV3Config = UltraV3Config(), verbose: bool = True) -> tuple[km.Circuit, dict]:
    t0 = time.time()
    info = compute_v3_tables(cfg)
    defer_c2_ovf = cfg.defer_c2_ovf and (info['c0lo_unique'] == 256)
    if cfg.defer_c2_ovf and not defer_c2_ovf and verbose:
        print(f'  [note] c0_lo_qrom_tab has {info["c0lo_unique"]}/256 unique -> falling back to defer_c2_ovf=False')
    if verbose:
        print(
            f'Tables ready in {time.time()-t0:.2f}s: rat={info["rat"]:.4f}, mm={info["mm"]:+.2e}, '
            f'plo_acc={info["plo_acc_bits"]}, c0loQ={info["c0_lo_qrom_bits"]}(uniq={info["c0lo_unique"]}), '
            f'w_c2={info["w_c2"]}, max_lo_bit={info["max_lo_bit"]}, r1_widths={info["r1_block_widths"]}',
            flush=True,
        )

    a_masks = gl.addr_bit_masks(8)
    chi_var_bits = cfg.w_hi - 1
    c0hi_masks = gl.int_tables_to_bit_masks(info['c0_hi_var'], chi_var_bits)
    c1_masks = gl.int_tables_to_bit_masks(info['c1_tab'], cfg.wc1)
    c0lo_bits = info['c0_lo_qrom_bits']
    c0lo_masks = gl.int_tables_to_bit_masks(info['c0_lo_qrom_tab'], c0lo_bits)
    c2_masks = gl.int_tables_to_bit_masks(info['c2_tab'], cfg.wc2)

    w_c2 = info['w_c2']
    n_c1_hi = cfg.wc1 - w_c2
    # Ensure n_c0lo_q1 + n_c1_hi >= 23 so c0_lo_rest can be computed after c2_fwd is HMR-deferred!
    min_q1_c0lo = max(0, 23 - n_c1_hi)
    n_c0lo_q1 = min(c0lo_bits, max(cfg.n_c0lo_q1, min_q1_c0lo))
    n_c0lo_rest = c0lo_bits - n_c0lo_q1

    b = km.CircuitBuilder()
    pool = up.PoolAllocator(b, [])
    stage_stats = []

    def _log(name: str):
        stage_stats.append((name, b.num_allocated_qubits, len(pool.avail)))

    # --- Stage 1: Normalize (+ optional Early Wave 0 with 6-bit linear c0_hi & s3==1 fusion!) & compact ---
    x = b.create_quantum_register(22, name='x')
    ccx_chi = 0
    if cfg.early_w0_s3:
        s_arr_w0 = up2.normalize_to_s(b, x)
        ccx_chi = _apply_6bit_linear_c0_hi_wave0(
            b, x, s_arr_w0, info, chi_var_bits, n_tmp=cfg.n_tmp_w0, seed=cfg.seed + 1,
        )
        nz = b.alloc_qubits(1)[0]
        s_hmr_bits = up2.compact_s_from_e(b, x, nz, s_arr_w0)
        _log(f'1_Norm+EarlyW0(ccx={ccx_chi})')
    else:
        nz = b.alloc_qubits(1)[0]
        s_hmr_bits = up2.normalize_and_compact(b, x, nz)
        _log('1_Norm')
    addr = list(x[14:22])
    u_c2 = list(x[14 - cfg.d_c2 : 14])

    # --- Stage 2: Zero-Ancilla GF(2) QROM1 for (c1, c0_lo_part) ---
    cqb = gl.compute_quad_bits_bcast if cfg.use_bcast else gl.compute_quad_bits
    c1 = list(b.alloc_qubits(cfg.wc1))
    c0_lo_part = list(b.alloc_qubits(n_c0lo_q1))
    tmp_q1 = list(b.alloc_qubits(cfg.n_temp_q1)) if cfg.n_temp_q1 > 0 else []
    q1_qs = c1 + c0_lo_part
    q1_ms = c1_masks + c0lo_masks[:n_c0lo_q1]
    if cfg.use_bcast:
        s_q1 = _pick_best_qrom1_seed(
            gl.synthesize_qrom1_bcast, 8, len(q1_qs), q1_ms, len(tmp_q1),
            cfg.seed, cfg.lut_seeds, n_boot=cfg.n_boot_q1,
        )
        ccx_q1 = gl.synthesize_qrom1_bcast(
            b, addr, q1_qs, q1_ms, temp_qubits=tmp_q1, n_boot=cfg.n_boot_q1, seed=s_q1
        )
    else:
        ccx_q1 = gl.synthesize_qrom1(
            b, addr, q1_qs, q1_ms, temp_qubits=tmp_q1, n_boot=cfg.n_boot_q1, seed=cfg.seed
        )
    if tmp_q1:
        b.free(km.array(tmp_q1))
    _log(f'2_QROM1(ccx={ccx_q1})')

    # --- Stage 3: Free `addr` (8 qubits) in 0 Toffolis via HMR+CZ -> PoolAllocator! ---
    gl.uncompute_bits_hmr(b, q1_qs, q1_ms, addr, a_masks)
    pool.add_pool(addr)
    _log('3_FreeAddr')

    # --- Stage 4: Wave 0 (`c0_hi`: 7 variable qubits + implicit MSB=1) — skipped if early_w0_s3! ---
    if not cfg.early_w0_s3:
        c0_hi = pool.alloc(chi_var_bits)
        tmp_chi = pool.alloc(cfg.n_boost_c0hi) if cfg.n_boost_c0hi > 0 else []
        ccx_chi = cqb(b, q1_qs, q1_ms, c0_hi, c0hi_masks, temp_qubits=tmp_chi, seed=cfg.seed + 1)
        if tmp_chi:
            pool.free(tmp_chi)

        s_hi = pool.alloc(3)
        s_arr = up2.unpack_s(b, x, nz, s_hi, s_hmr_bits=s_hmr_bits, alloc=pool)
        up2.streaming_phase_shift_reuse(
            b, c0_hi, list(s_arr), [x[0], x[2], x[3]], info['angs_hi'],
            max_s=10, implicit_msb=True, alloc=pool,
        )
        up2.pack_s(b, x, nz, s_hi, alloc=pool)
        pool.free(s_hi)
        gl.uncompute_bits_hmr(b, q1_qs, q1_ms, c0_hi, c0hi_masks)
        pool.free(c0_hi)
        s_hmr_bits = None  # already consumed in Stage 4!
        _log(f'4_Wave0(ccx={ccx_chi})')

    # --- Stage 5: Unsplit `c2` + `c2 * u_c2` (with deferred HMR!) + `c0_lo_rest` (keeping c2 alive!) ---
    c2_q = pool.alloc(cfg.wc2)
    n_bst_c2f = min(cfg.n_boost_c2fwd, max(0, 60 - b.num_allocated_qubits + len(pool.avail)))
    tmp_c2f = pool.alloc(n_bst_c2f) if n_bst_c2f > 0 else []
    s_c2f = _pick_best_lut_seed(cqb, len(q1_qs), q1_ms, len(c2_q), c2_masks, len(tmp_c2f), cfg.seed + 2, cfg.lut_seeds)
    ccx_c2fwd = cqb(b, q1_qs, q1_ms, c2_q, c2_masks, temp_qubits=tmp_c2f, seed=s_c2f)
    if tmp_c2f:
        pool.free(tmp_c2f)

    c2_ovf = pool.alloc(1)[0]
    c2_tokens: dict = {}
    _apply_c2_unsplit_1ovf(
        b, False, u_c2, c2_q, c1[:w_c2], c2_ovf,
        info['c2_rows'], info['sh_c2'], max_anc=cfg.c2_max_anc,
        alloc=pool, defer_tokens=c2_tokens, mux_round=cfg.mux_round_c2,
    )

    # Keep `n_keep_c2` pure qubits of `c2_q` alive while computing `c0_lo_rest` (gives ~30 pure wires!),
    # and HMR-defer the rest first to keep peak Q <= 60!
    n_keep_c2 = min(cfg.wc2, max(0, cfg.n_c2_keep_c0r)) if n_c0lo_rest > 0 else 0
    b_c2_fwd: list[km.b] = [None] * cfg.wc2  # type: ignore
    if n_keep_c2 < cfg.wc2:
        drop_bits = gl.hmr_defer_bits(b, c2_q[n_keep_c2:])
        for k_i, mb in enumerate(drop_bits):
            b_c2_fwd[n_keep_c2 + k_i] = mb
        pool.free(c2_q[n_keep_c2:])
    _log(f'5a_c2fwd(ccx={ccx_c2fwd})')

    # Compute `c0_lo_rest` using pure `c0_lo_part + untouched c1[w_c2:] + c2_q[:n_keep_c2]` (~30 pure wires!)
    pure_c0r_q = c0_lo_part + c1[w_c2:] + c2_q[:n_keep_c2]
    pure_c0r_m = c0lo_masks[:n_c0lo_q1] + c1_masks[w_c2:] + c2_masks[:n_keep_c2]
    c0_lo_rest = pool.alloc(n_c0lo_rest)
    ccx_c0r = 0
    if n_c0lo_rest > 0:
        tmp_c0r = pool.alloc(cfg.n_boost_c0r) if cfg.n_boost_c0r > 0 else []
        s_c0r = _pick_best_lut_seed(
            cqb, len(pure_c0r_q), pure_c0r_m, len(c0_lo_rest), c0lo_masks[n_c0lo_q1:],
            len(tmp_c0r), cfg.seed + 3, cfg.lut_seeds,
        )
        ccx_c0r = cqb(
            b, pure_c0r_q, pure_c0r_m, c0_lo_rest, c0lo_masks[n_c0lo_q1:],
            temp_qubits=tmp_c0r, seed=s_c0r,
        )
        if tmp_c0r:
            pool.free(tmp_c0r)
    if n_keep_c2 > 0:
        keep_bits = gl.hmr_defer_bits(b, c2_q[:n_keep_c2])
        for k_i, mb in enumerate(keep_bits):
            b_c2_fwd[k_i] = mb
        pool.free(c2_q[:n_keep_c2])
    c0_lo = c0_lo_part + c0_lo_rest

    # Propagate `c2_ovf` into `c1[w_c2:]` (with deferred HMR checkpoints!)
    c2_prop_tok: dict = {}
    i0_c2 = info['c2_rows'][0]
    up2.propagate_1ovf_carry(
        b, False, u_c2[i0_c2], True, c2_ovf, c1[w_c2:],
        max_anc=cfg.c2_max_anc, alloc=pool, defer_hmr=c2_prop_tok,
    )
    b_c2ovf_fwd: km.b | None = None
    if defer_c2_ovf:
        b_c2ovf_fwd = gl.hmr_defer_bits(b, [c2_ovf])[0]
        pool.free(c2_ovf)
    _log(f'5b_c0r+prop(ccx={ccx_c0r})')

    # --- Stage 6: `r1 * u` (Deferred-HMR forward, 0 conditional CCXs, packed `s`!) ---
    p_lo_gb = pool.alloc(cfg.gb) if (cfg.zgb and cfg.gb > 0) else []
    if cfg.zgb and cfg.gb > 0:
        for bi in range(cfg.gb):
            if (info['gb_const'] >> bi) & 1:
                b.x(p_lo_gb[bi])
    p_lo_acc = p_lo_gb + c0_lo

    r1_grps = info['r1_grps']
    r1_widths = info['r1_block_widths']
    r1_ovfs: list[km.q] = []
    r1_tokens: dict = {}

    for bi_idx, (grp, wb) in enumerate(zip(r1_grps[:-1], r1_widths[:-1])):
        ma = max(4, cfg.r1_anc_budget - len(r1_ovfs))
        if cfg.cap_r1_anc_to_pool:
            # Forward defer_hmr uses 0 cond CCX at ANY max_anc >= 2! Cap to pool_avail (reserving 1 for ovf if needed).
            ma = min(ma, max(2, len(pool.avail) - (1 if wb is not None else 0)))
        blk_tok = r1_tokens.setdefault(bi_idx, {})
        if wb is None:
            up2.signed_digit_block_mul_1ovf(
                b, False, x[:14], c1, p_lo_acc, [],
                shift_base=info['sh_r1'], blocks=[(grp, None)], max_anc=ma,
                subtract=True, lazy_ovf=False, alloc=pool, defer_tokens=blk_tok,
                mux_round=cfg.mux_round_r1,
            )
        else:
            ovf_q = pool.alloc(1)[0]
            r1_ovfs.append(ovf_q)
            up2.signed_digit_block_mul_1ovf(
                b, False, x[:14], c1, p_lo_acc, [ovf_q],
                shift_base=info['sh_r1'], blocks=[(grp, wb)], max_anc=ma,
                subtract=True, lazy_ovf=False, alloc=pool, defer_tokens=blk_tok,
                mux_round=cfg.mux_round_r1,
            )
    _log(f'6a_r1uTop(ovfs={len(r1_ovfs)})')

    last_grp, last_wb = r1_grps[-1], r1_widths[-1]
    assert last_wb is not None
    ovf_last = pool.alloc(1)[0]
    r1_ovfs.append(ovf_last)
    ma_last = max(4, cfg.r1_anc_budget - len(r1_ovfs) + 1)
    ma_last_fwd = min(ma_last, max(2, len(pool.avail))) if cfg.cap_r1_anc_to_pool else ma_last
    if cfg.pack_s_in_r1B and last_grp[0] >= 4:
        _apply_r1_packed_rows(
            b, False, x, nz, c1, p_lo_acc, ovf_last, last_grp, last_wb, info['sh_r1'],
            max_anc=ma_last_fwd, alloc=pool, defer_tokens=r1_tokens, hmr_ovf=True,
            mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_mux_23=cfg.skip_mux_23, blk_key='last',
        )
        s_hi2 = pool.alloc(3)
        s_arr2 = up2.unpack_s(b, x, nz, s_hi2, s_hmr_bits=s_hmr_bits, alloc=pool)
    else:
        s_hi2 = pool.alloc(3)
        s_arr2 = up2.unpack_s(b, x, nz, s_hi2, s_hmr_bits=s_hmr_bits, alloc=pool)
        up2.signed_digit_block_mul_1ovf(
            b, False, x[:14], c1, p_lo_acc, [ovf_last],
            shift_base=info['sh_r1'], blocks=[(last_grp, last_wb)], max_anc=ma_last_fwd,
            subtract=True, lazy_ovf=False, alloc=pool, defer_tokens=r1_tokens.setdefault('last', {}),
            mux_round=cfg.mux_round_r1,
        )
    _log(f'6b_r1uBot(ovfs={len(r1_ovfs)})')

    # --- Stage 7: Wave 1 (`p_lo = p_lo_acc[gb:]`, early stop at `max_lo_bit`!) ---
    p_lo = p_lo_acc[cfg.gb:]
    up2.streaming_phase_shift_reuse(
        b, p_lo, list(s_arr2), [x[0], x[2], x[3]], info['angs_lo'],
        max_s=10, implicit_msb=False, allow_early_stop=True, alloc=pool,
        s3_max_bits=(info['s3_cut'] if cfg.early_w0_s3 else None),
    )
    _log('7_Wave1')

    # --- Stage 8: Uncompute `r1 * u` (with asymmetric deferred-HMR replay + `HMR(ovf)`!) ---
    if cfg.pack_s_in_r1B and last_grp[0] >= 4:
        up2.pack_s(b, x, nz, s_hi2, alloc=pool)
        pool.free(s_hi2)
        ovf_q = r1_ovfs.pop()
        # Note: `_apply_r1_packed_rows(..., hmr_ovf=True)` frees `ovf_q` before reverse rows (+1 pool_avail)!
        ma_last_rev = min(ma_last, max(2, len(pool.avail))) if cfg.cap_r1_anc_to_pool else ma_last
        _apply_r1_packed_rows(
            b, True, x, nz, c1, p_lo_acc, ovf_q, last_grp, last_wb, info['sh_r1'],
            max_anc=ma_last_rev, alloc=pool, defer_tokens=r1_tokens, hmr_ovf=True,
            mux_round=cfg.mux_round_r1, mux_min_row=cfg.mux_r1_min_row, skip_mux_23=cfg.skip_mux_23, blk_key='last',
        )
    else:
        ovf_q = r1_ovfs.pop()
        ma_last_rev = min(ma_last, max(2, len(pool.avail))) if cfg.cap_r1_anc_to_pool else ma_last
        up2.signed_digit_block_mul_1ovf(
            b, True, x[:14], c1, p_lo_acc, [ovf_q],
            shift_base=info['sh_r1'], blocks=[(last_grp, last_wb)], max_anc=ma_last_rev,
            subtract=True, lazy_ovf=False, alloc=pool, defer_tokens=r1_tokens['last'],
            hmr_ovf=True, mux_round=cfg.mux_round_r1,
        )
        pool.free(ovf_q)
        up2.pack_s(b, x, nz, s_hi2, alloc=pool)
        pool.free(s_hi2)

    for bi_idx in range(len(r1_grps) - 2, -1, -1):
        grp, wb = r1_grps[bi_idx], r1_widths[bi_idx]
        blk_tok = r1_tokens[bi_idx]
        if wb is None:
            ma = max(4, cfg.r1_anc_budget - len(r1_ovfs))
            if cfg.cap_r1_anc_to_pool:
                ma = min(ma, max(2, len(pool.avail)))
            up2.signed_digit_block_mul_1ovf(
                b, True, x[:14], c1, p_lo_acc, [],
                shift_base=info['sh_r1'], blocks=[(grp, None)], max_anc=ma,
                subtract=True, lazy_ovf=False, alloc=pool, defer_tokens=blk_tok,
                mux_round=cfg.mux_round_r1,
            )
        else:
            ovf_q = r1_ovfs.pop()
            # With lazy_ovf=True + hmr_ovf=True, signed_digit_block_mul_1ovf frees `ovf_q` BEFORE reverse rows (+1 anc)!
            ma = max(4, cfg.r1_anc_budget - len(r1_ovfs))
            if cfg.cap_r1_anc_to_pool:
                ma = min(ma, max(2, len(pool.avail) + 1))
            up2.signed_digit_block_mul_1ovf(
                b, True, x[:14], c1, p_lo_acc, [ovf_q],
                shift_base=info['sh_r1'], blocks=[(grp, wb)], max_anc=ma,
                subtract=True, lazy_ovf=True, alloc=pool, defer_tokens=blk_tok,
                hmr_ovf=True, mux_round=cfg.mux_round_r1,
            )

    if cfg.zgb and cfg.gb > 0:
        for bi in range(cfg.gb):
            if (info['gb_const'] >> bi) & 1:
                b.x(p_lo_gb[bi])
        pool.free(p_lo_gb)
    _log('8_UncompR1u')

    # --- Stage 9: Uncompute `c2 * u_c2` (recover `c2_ovf`, early `HMR(c0_lo)` -> unpebbled rev `c2*u`!) ---
    ccx_c2ovf = 0
    if defer_c2_ovf:
        c2_ovf_r = pool.alloc(1)[0]
        # Compute c1_orig[w_c2] from pure `c0_lo` + init_and temp boost (5-6 wires -> M >= 25!)
        n_t_ovf = max(4, 26 - len(c0_lo))
        tmp_ovf = pool.alloc(n_t_ovf)
        ccx_c2ovf = cqb(
            b, c0_lo, c0lo_masks, [c2_ovf_r], [c1_masks[w_c2]],
            temp_qubits=tmp_ovf, seed=cfg.seed + 4, use_init_and_temp=True,
        )
        pool.free(tmp_ovf)
        # c2_ovf = c1_orig[w_c2] ^ r1[w_c2] ^ u_c2[i0_c2]
        b.cx(c1[w_c2], c2_ovf_r)
        b.cx(u_c2[i0_c2], c2_ovf_r)
        assert b_c2ovf_fwd is not None
        b.push_condition(b_c2ovf_fwd)
        b.z(c2_ovf_r)
        b.pop_condition()
        b.free(b_c2ovf_fwd)
    else:
        c2_ovf_r = c2_ovf

    # Undo carry propagation into c1[w_c2:] -> c1[w_c2:] is now 100% PURE again!
    up2.propagate_1ovf_carry(
        b, True, u_c2[i0_c2], True, c2_ovf_r, c1[w_c2:],
        max_anc=cfg.c2_max_anc, alloc=pool, defer_hmr=c2_prop_tok,
    )
    # HMR-free c2_ovf_r right now -> 0-Tof phase_cout_bit during reverse c2*u (-1 CCX/row!)
    b_c2ovf_rev = gl.hmr_defer_bits(b, [c2_ovf_r])[0]
    pool.free(c2_ovf_r)

    # Free `n_drop_c0` bits of `c0_lo` BEFORE computing `c2_rev` (keeping `len(c0_lo) - n_drop_c0 + n_c1_hi >= 23`!)
    max_drop_c0 = max(0, len(c0_lo) + n_c1_hi - 23)
    n_drop_c0 = min(max_drop_c0, max(n_c0lo_rest, cfg.n_c0_drop_c2r))
    # Always drop `c0_lo_rest` first (which came from pool), plus any extra tail bits of `c0_lo_part`!
    c0_drop_q = c0_lo[len(c0_lo) - n_drop_c0 :] if n_drop_c0 > 0 else []
    c0_keep_q = c0_lo[: len(c0_lo) - n_drop_c0]
    b_c0lo_all: list[km.b] = [None] * len(c0_lo)  # type: ignore
    # Clean up c0_lo_rest qubits that came from builder (not in addr or c0_lo_part!)
    c0_lo_part_set = set(c0_lo_part)
    extra_c0r_builder = [q for q in c0_lo_rest if q not in pool.pool_set]
    if c0_drop_q:
        db = gl.hmr_defer_bits(b, c0_drop_q)
        base_i = len(c0_lo) - n_drop_c0
        for k_i, mb in enumerate(db):
            b_c0lo_all[base_i + k_i] = mb
        for q in c0_drop_q:
            if q in pool.pool_set:
                pool.free(q)
            else:
                pool.add_pool([q])

    # Recompute unsplit `c2_rev` (8 bits) from `c0_keep_q + c1[w_c2:]` (>= 23 pure wires) + boost + BCAST!
    pure_c2r_q = c0_keep_q + c1[w_c2:]
    pure_c2r_m = c0lo_masks[: len(c0_keep_q)] + c1_masks[w_c2:]
    c2_rev_q = pool.alloc(cfg.wc2)
    tmp_c2r = pool.alloc(cfg.n_boost_c2rev) if cfg.n_boost_c2rev > 0 else []
    s_c2r = _pick_best_lut_seed(
        cqb, len(pure_c2r_q), pure_c2r_m, len(c2_rev_q), c2_masks, len(tmp_c2r), cfg.seed + 5, cfg.lut_seeds,
    )
    ccx_c2rev = cqb(
        b, pure_c2r_q, pure_c2r_m, c2_rev_q, c2_masks,
        temp_qubits=tmp_c2r, seed=s_c2r,
    )
    if tmp_c2r:
        pool.free(tmp_c2r)

    # Immediately HMR-defer remaining `c0_keep_q` -> +15+ FREE ANCILLAS in pool for reverse c2*u!
    if c0_keep_q:
        kb = gl.hmr_defer_bits(b, c0_keep_q)
        for k_i, mb in enumerate(kb):
            b_c0lo_all[k_i] = mb
        for q in c0_keep_q:
            if q in pool.pool_set:
                pool.free(q)
            else:
                pool.add_pool([q])

    # Run reverse unsplit `c2 * u_c2` with `hmr_ovf` (phase_cout_bit=b_c2ovf_rev) and plenty of pool ancillas!
    _apply_c2_unsplit_1ovf(
        b, True, u_c2, c2_rev_q, c1[:w_c2], None,
        info['c2_rows'], info['sh_c2'], max_anc=12,
        alloc=pool, defer_tokens=c2_tokens, phase_cout_bit=b_c2ovf_rev,
        mux_round=cfg.mux_round_c2,
    )
    b.free(b_c2ovf_rev)
    _log(f'9_UncompC2(ovf={ccx_c2ovf},c2r={ccx_c2rev})')

    # --- Stage 10: Restore `addr` -> 0-Tof deferred CZ fixups -> `init_and` uncompute `c1`! ---
    # Live pure wires: `c1` (18) + `c2_rev_q` (8) = 26 pure wires! Plus 15+ free qubits in `pool`!
    addr_tmp = pool.alloc(8)
    n_bst_ar = min(cfg.n_boost_addr, len(pool.avail))
    tmp_ar = pool.alloc(n_bst_ar) if n_bst_ar > 0 else []
    s_ar = _pick_best_lut_seed(
        cqb, len(c1) + len(c2_rev_q), c1_masks + c2_masks, 8, a_masks, len(tmp_ar), cfg.seed + 6, cfg.lut_seeds,
    )
    ccx_addr_r = cqb(
        b, c1 + c2_rev_q, c1_masks + c2_masks, addr_tmp, a_masks,
        temp_qubits=tmp_ar, seed=s_ar,
    )
    if tmp_ar:
        pool.free(tmp_ar)

    # Free `c2_rev_q` (0 Tof HMR+CZ using addr_tmp + c1 = 26 wires!) -> frees any `addr` qubits held by c2_rev_q!
    gl.uncompute_bits_hmr(b, addr_tmp + c1, a_masks + c1_masks, c2_rev_q, c2_masks)
    pool.free(c2_rev_q)

    # Move `addr_tmp` -> exact `addr` register via 0-Toffoli SWAPs (handling any partial overlap cleanly!)
    cur_loc = list(addr_tmp)
    for j in range(8):
        if cur_loc[j] != addr[j]:
            q_src = cur_loc[j]
            q_dst = addr[j]
            b.swap(q_src, q_dst)
            if q_dst in cur_loc[j + 1 :]:
                k = cur_loc.index(q_dst, j + 1)
                cur_loc[k] = q_src
            else:
                pool.avail.remove(q_dst)
                pool.pool_set.remove(q_dst)
                if q_src in pool.pool_set:
                    pool.avail.append(q_src)
                else:
                    b.free(q_src)
            cur_loc[j] = q_dst
        else:
            pool.pool_set.discard(addr[j])

    # Apply ALL deferred 0-Toffoli CZ phase fixups using `addr + c1` (26 pure wires -> 0 CCX!)
    gl.apply_deferred_quad_fixup(b, addr + c1, a_masks + c1_masks, c2_masks, b_c2_fwd)
    gl.apply_deferred_quad_fixup(b, addr + c1, a_masks + c1_masks, c0lo_masks, b_c0lo_all)

    # Uncompute `c1` (17-18 bits) using greedy `init_and`/`del_and` (~16-17 CCX vs 25 CCX del_lookup!)
    n_hmr_q1, n_ia_q1 = gl.uncompute_qrom1_init_and(b, addr, c1, c1_masks, alloc=pool, seed=cfg.seed + 7)
    pool.remove_pool(c0_lo_part)
    if extra_c0r_builder:
        pool.remove_pool(extra_c0r_builder)
        b.free(km.array(extra_c0r_builder))
    b.free(km.array(c0_lo_part))
    b.free(km.array(c1))
    _log(f'10_UncompQ1(ar={ccx_addr_r},hmr={n_hmr_q1},ia={n_ia_q1})')

    # --- Stage 11: Unnormalize x from compact form & free nz! ---
    up2.unnormalize_from_compact(b, x, nz)
    b.free(nz)
    _log('11_Unnorm')

    circ = b.finish_circuit()
    info['ccx_breakdown'] = {
        'qrom1': ccx_q1, 'c0_hi': ccx_chi, 'c0lo_rest': ccx_c0r,
        'c2_fwd': ccx_c2fwd, 'c2_ovf_r': ccx_c2ovf, 'c2_rev': ccx_c2rev,
        'addr_rev': ccx_addr_r, 'hmr_q1': n_hmr_q1, 'ia_q1': n_ia_q1,
    }
    info['stage_stats'] = stage_stats
    if verbose:
        st = circ.operation_statistics()
        static_tof = st.get('CCX', 0) + st.get('CSWAP', 0) + st.get('CCZ', 0)
        print(f'Built ultra_v3 in {time.time()-t0:.2f}s: Q={circ.num_qubits}, static_Tof={static_tof}, Z_POW={st.get("Z_POW", 0)}', flush=True)
        print('  CCX breakdown:', info['ccx_breakdown'], flush=True)
        print('  Qubit peaks (name, alloc, pool_avail):', stage_stats, flush=True)
    return circ, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output', nargs='?', default='ultra_v3_a.kmb')
    ap.add_argument('--wc1', type=int, default=18)
    ap.add_argument('--wc2', type=int, default=8)
    ap.add_argument('--dc2', type=int, default=8)
    ap.add_argument('--wp', type=int, default=26)
    ap.add_argument('--gb', type=int, default=1)
    ap.add_argument('--zgb', type=int, default=0)
    ap.add_argument('--whi', type=int, default=8)
    ap.add_argument('--skip37', type=int, default=0)
    ap.add_argument('--mux-r1', type=int, default=1)
    ap.add_argument('--mux-c2', type=int, default=0)
    ap.add_argument('--nc0q1', type=int, default=15)
    ap.add_argument('--ntmpq1', type=int, default=4)
    ap.add_argument('--nbootq1', type=int, default=11)
    ap.add_argument('--r1-blk', type=str, default='2blk')
    ap.add_argument('--r1-anc', type=int, default=7)
    ap.add_argument('--c2-anc', type=int, default=2)
    ap.add_argument('--mux-min', type=int, default=0)
    ap.add_argument('--defer-c2ovf', type=int, default=1)
    ap.add_argument('--early-w0', type=int, default=1)
    ap.add_argument('--cap-r1', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    cfg = UltraV3Config(
        wc1=args.wc1, wc2=args.wc2, d_c2=args.dc2, wp=args.wp, gb=args.gb, zgb=args.zgb,
        w_hi=args.whi, skip37=bool(args.skip37), mux_round_r1=bool(args.mux_r1),
        mux_r1_min_row=args.mux_min, mux_round_c2=bool(args.mux_c2),
        n_c0lo_q1=args.nc0q1, n_temp_q1=args.ntmpq1, n_boot_q1=args.nbootq1,
        r1_block_mode=args.r1_blk, r1_anc_budget=args.r1_anc, c2_max_anc=args.c2_anc,
        defer_c2_ovf=bool(args.defer_c2ovf), early_w0_s3=bool(args.early_w0),
        cap_r1_anc_to_pool=bool(args.cap_r1), seed=args.seed,
    )
    circ, _ = build_ultra_v3(cfg, verbose=True)
    circ.to_file(args.output)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
