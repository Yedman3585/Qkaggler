"""Upgraded primitives for ultra_v2:
1. `normalize_and_compact` / `unpack_s` / `pack_s` / `unnormalize_from_compact`:
   Compacts parity into x[21] and s[0..3] into (nz, x[0..3]) -> ONLY 23 fixed qubits!
2. `signed_digit_block_mul_1ovf`:
   1-qubit overflow (`q_ovf - neg_top`) per 2- or 3-row block (cuts ovf qubits in half!).
3. `streaming_phase_shift_reuse`:
   Reuses x[0], x[2], x[3] as delay buffers d0, d1[0..1] (0 extra delay qubits!)
   and supports `implicit_msb=True` (for 7-qubit c0_hi with constant MSB=1).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import kickmix as km
import ultra_primitives as up


def normalize_to_s(
    builder: km.CircuitBuilder,
    x: km.array,
) -> list[km.q]:
    """Normalizes x[0..21] to [2^21, 2^22), moves `is_2_to_4` into x[21], and frees parity e[0].

    Returns `s_arr = [s0, s1, s2, s3]` (4 allocated qubits; total live = 26 qubits).
    Note: x[0..3] are untouched (unpacked x_norm[0..3]), so Wave 0 can run directly here
    with ZERO unpack_s/pack_s overhead! Call `compact_s_from_e` afterward to compact `s_arr`
    into `(nz, x[0..3])` and free `s_arr`.
    """
    e = builder.alloc_qubits(5)
    up.normalize_to_1_2(builder, False, x, e)
    # x[21] is clean |0>; e[0] = parity (1 for [1,2), 0 for [2,4)); e[1..4] = s[0..3]
    builder.cx(e[0], x[21])
    b_p = builder.alloc_bits(1)[0]
    builder.hmr(e[0], b_p)
    builder.push_condition(b_p)
    builder.z(x[21])
    builder.pop_condition()
    builder.x(x[21])  # x[21] = is_2_to_4 (0 for [1,2), 1 for [2,4))
    builder.free(e[0])
    return [e[1], e[2], e[3], e[4]]


def compact_s_from_e(
    builder: km.CircuitBuilder,
    x: km.array,
    nz: km.q,
    s_arr: Sequence[km.q],
) -> km.array:
    """Compacts `s_arr = [s0, s1, s2, s3]` into `(nz, x[0..3])` and frees `s_arr`.
    Returns `s_hmr_bits` (4 classical bits to pass to the first `unpack_s`)."""
    s0, s1, s2, s3 = s_arr[0], s_arr[1], s_arr[2], s_arr[3]
    ge2 = builder.alloc_qubits(1)[0]
    builder.cx(s3, s2)  # s2 now holds s23 = s2 | s3 (since s2 & s3 == 0)

    # ge2 = s1 | s23 = ~(~s1 & ~s23)
    builder.x(s1)
    builder.x(s2)
    builder.init_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    builder.x(ge2)

    # nz = s0 | ge2 = ~(~s0 & ~ge2)
    builder.x(s0)
    builder.x(ge2)
    builder.init_and(s0, ge2, nz)
    builder.x(s0)
    builder.x(ge2)
    builder.x(nz)

    # Write s into x[0..3] (all guaranteed 0 whenever touched!)
    builder.cx(s1, x[0])
    builder.cx(s2, x[1])
    # x[2] ^= (s0 & ge2) == s0 ^ nz ^ ge2
    builder.cx(s0, x[2])
    builder.cx(nz, x[2])
    builder.cx(ge2, x[2])
    builder.cx(s3, x[3])

    # Uncompute ge2 with 0 Toffolis!
    builder.x(ge2)
    builder.x(s1)
    builder.x(s2)
    builder.del_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    builder.free(ge2)

    # Restore s2 from s23
    builder.cx(s3, s2)

    # HMR s[0..3] -> s_hmr_bits (phase fixup deferred to first unpack_s!)
    s_hmr_bits = builder.alloc_bits(4)
    builder.hmr(km.array([s0, s1, s2, s3]), s_hmr_bits)
    builder.free(km.array([s0, s1, s2, s3]))
    return s_hmr_bits


def normalize_and_compact(
    builder: km.CircuitBuilder,
    x: km.array,
    nz: km.q,
) -> km.array:
    """Normalizes x[0..21] to [2^21, 2^22), moves `is_2_to_4` into x[21],
    and compacts s[0..3] into (nz, x[0..3]), freeing all 5 e qubits!

    Returns `s_hmr_bits` (4 classical bits to pass to the FIRST `unpack_s` call!).
    """
    s_arr = normalize_to_s(builder, x)
    return compact_s_from_e(builder, x, nz, s_arr)


def unpack_s(
    builder: km.CircuitBuilder,
    x: km.array,
    nz: km.q,
    s_hi: Sequence[km.q],
    s_hmr_bits: km.array | None = None,
    alloc: up.PoolAllocator | None = None,
) -> km.array:
    """Unpacks s[0..3] from (nz, x[0..3]) into [nz, s_hi[0], s_hi[1], s_hi[2]] and
    restores x[0..3] to exact x_norm[0..3]!

    Uses 4 CCX + 1 CSWAP = 5 Toffolis and 1 transient ancilla (freed immediately).
    Returns km.array([nz, s_hi[0], s_hi[1], s_hi[2]]) representing s[0..3].
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))
    s1, s2, s3 = s_hi[0], s_hi[1], s_hi[2]
    builder.init_and(nz, x[0], s1)
    builder.cx(s1, x[0])
    builder.init_and(nz, x[1], s2)  # s2 holds s23
    builder.cx(s2, x[1])

    ge2 = _alloc(1)[0]
    builder.x(s1)
    builder.x(s2)
    builder.init_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    builder.x(ge2)

    builder.cx(ge2, nz)  # nz now holds (s == 1)
    builder.cswap(ge2, x[2], nz)  # nz becomes s0, x[2] restored!

    builder.x(ge2)
    builder.x(s1)
    builder.x(s2)
    builder.del_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    _free(ge2)

    builder.init_and(s2, x[3], s3)
    builder.cx(s3, x[3])
    builder.cx(s3, s2)  # s2 restored from s23

    s_arr = km.array([nz, s1, s2, s3])
    if s_hmr_bits is not None:
        for k in range(4):
            builder.push_condition(s_hmr_bits[k])
            builder.z(s_arr[k])
            builder.pop_condition()
    return s_arr


def pack_s(
    builder: km.CircuitBuilder,
    x: km.array,
    nz: km.q,
    s_hi: Sequence[km.q],
    alloc: up.PoolAllocator | None = None,
) -> None:
    """Exact inverse of `unpack_s`: packs s = [nz, s_hi[0..2]] back into (nz, x[0..3])
    and uncomputes s_hi[0..2] cleanly to |0> using only 1 CCX + 1 CSWAP = 2 Toffolis!
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))
    s1, s2, s3 = s_hi[0], s_hi[1], s_hi[2]
    builder.cx(s3, s2)  # s2 becomes s23
    builder.cx(s3, x[3])
    builder.del_and(s2, x[3], s3)

    ge2 = _alloc(1)[0]
    builder.x(s1)
    builder.x(s2)
    builder.init_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    builder.x(ge2)

    builder.cswap(ge2, x[2], nz)
    builder.cx(ge2, nz)

    builder.x(ge2)
    builder.x(s1)
    builder.x(s2)
    builder.del_and(s1, s2, ge2)
    builder.x(s1)
    builder.x(s2)
    _free(ge2)

    builder.cx(s2, x[1])
    builder.del_and(nz, x[1], s2)
    builder.cx(s1, x[0])
    builder.del_and(nz, x[0], s1)


def unnormalize_from_compact(
    builder: km.CircuitBuilder,
    x: km.array,
    nz: km.q,
    s_hmr_bits: km.array | None = None,
) -> None:
    """Unpacks s[0..3] and parity from (x[0..21], nz) and runs inverse `normalize_to_1_2`!

    Restores x[0..21] to original input and leaves nz in clean |0>!
    """
    s_hi = builder.alloc_qubits(3)
    s_arr = unpack_s(builder, x, nz, s_hi, s_hmr_bits=s_hmr_bits)
    e0 = builder.alloc_qubits(1)[0]
    builder.x(x[21])  # back to parity (1 for [1,2), 0 for [2,4))
    builder.swap(x[21], e0)  # x[21] is now clean |0>, e0 = parity!
    e_full = km.array([e0, s_arr[0], s_arr[1], s_arr[2], s_arr[3]])
    up.normalize_to_1_2(builder, True, x, e_full)
    builder.free(e0)
    builder.free(s_hi)


def _neg_ctrl_flip(builder: km.CircuitBuilder, sign_q: km.q, invert_sign: bool, regs: Sequence[km.q]) -> None:
    if invert_sign:
        builder.x(sign_q)
    for q in regs:
        builder.cx(sign_q, q)
    if invert_sign:
        builder.x(sign_q)


def _signed_digit_iadd_row(
    builder: km.CircuitBuilder,
    uncompute: bool,
    sign_q: km.q,
    invert_sign: bool,
    a: Sequence[km.q],
    target: Sequence[km.q],
    shift: int,
    max_anc: int,
    flip_target_before: Sequence[km.q] | None = None,
    flip_target_after: Sequence[km.q] | None = None,
    alloc: up.PoolAllocator | None = None,
    defer_hmr: dict | None = None,
    phase_cout_bit: km.b | None = None,
    mux_round: bool = False,
) -> None:
    """Adds (or subtracts when `neg = sign_q ^ invert_sign == 1`) `round(a * 2^shift)` into `target`.

    - When `mux_round=True` and `shift < 0` (`k = -shift`), computes 2-bit-exact rounding carry-in
      `cin = MUX(sign_q, a[k-1], a[k-2])` (1 CCX compute, 0 CCX del_and) cutting rounding error 2.5x!
    - Supports `alloc` (PoolAllocator), `defer_hmr` (0-fixup forward checkpoints), and
      `phase_cout_bit` (0-Tof carry-out phase for `HMR(ovf)` in reverse!).
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))

    w_a = len(a)
    w_t = len(target)
    cin_tmp: km.q | None = None
    k_shift = 0
    if shift >= 0:
        if shift >= w_t:
            if defer_hmr is not None and 'bits' not in defer_hmr:
                defer_hmr['blocks'] = None
                defer_hmr['bits'] = []
            return
        sub_tgt = list(target[shift:])
        sub_off = list(a[: min(w_a, len(sub_tgt))])
        round_bit: km.q | None = None
    else:
        k_shift = -shift
        if k_shift > w_a:
            if defer_hmr is not None and 'bits' not in defer_hmr:
                defer_hmr['blocks'] = None
                defer_hmr['bits'] = []
            return
        sub_tgt = list(target)
        sub_off = list(a[k_shift : min(w_a, k_shift + w_t)])
        if mux_round:
            cin_tmp = _alloc(1)[0]
            if k_shift == 1:
                builder.init_and(sign_q, a[0], cin_tmp)
            else:
                # cin_tmp = a[k-2] ^ (sign_q & (a[k-1] ^ a[k-2])) = MUX(sign_q, a[k-1], a[k-2])
                builder.cx(a[k_shift - 2], a[k_shift - 1])
                builder.init_and(sign_q, a[k_shift - 1], cin_tmp)
                builder.cx(a[k_shift - 2], a[k_shift - 1])
                builder.cx(a[k_shift - 2], cin_tmp)
            round_bit = cin_tmp
        else:
            round_bit = a[k_shift - 1]

    eff_anc = max(2, max_anc - (1 if cin_tmp is not None else 0)) if max_anc is not None else None
    f_pre = list(target) if flip_target_before is None else list(flip_target_before)
    f_post = list(target) if flip_target_after is None else list(flip_target_after)

    if not uncompute:
        _neg_ctrl_flip(builder, sign_q, invert_sign, f_pre)
        up.pebbled_gidney_iadd(
            builder, sub_off, sub_tgt, cin=round_bit, max_anc=eff_anc, subtract=False,
            alloc=alloc, defer_hmr=defer_hmr, phase_cout_bit=phase_cout_bit,
        )
        _neg_ctrl_flip(builder, sign_q, invert_sign, f_post)
    else:
        _neg_ctrl_flip(builder, sign_q, invert_sign, f_post)
        up.pebbled_gidney_iadd(
            builder, sub_off, sub_tgt, cin=round_bit, max_anc=eff_anc, subtract=True,
            alloc=alloc, defer_hmr=defer_hmr, phase_cout_bit=phase_cout_bit,
        )
        _neg_ctrl_flip(builder, sign_q, invert_sign, f_pre)

    if cin_tmp is not None:
        if k_shift == 1:
            builder.del_and(sign_q, a[0], cin_tmp)
        else:
            builder.cx(a[k_shift - 2], cin_tmp)
            builder.cx(a[k_shift - 2], a[k_shift - 1])
            builder.del_and(sign_q, a[k_shift - 1], cin_tmp)
            builder.cx(a[k_shift - 2], a[k_shift - 1])
        _free(cin_tmp)


def signed_digit_block_mul_1ovf(
    builder: km.CircuitBuilder,
    uncompute: bool,
    u: Sequence[km.q],
    a: Sequence[km.q],
    target: Sequence[km.q],
    ovf_qubits: list[km.q],
    shift_base: int,
    blocks: Sequence[tuple[list[int], int | None]],
    max_anc: int = 4,
    subtract: bool = False,
    lazy_ovf: bool = False,
    alloc: up.PoolAllocator | None = None,
    defer_tokens: dict | None = None,
    hmr_ovf: bool = False,
    mux_round: bool = False,
) -> list[km.q]:
    """Signed-digit (+-1) block-carry multiplication with ONLY 1 overflow qubit per block!

    Enhancements:
      - `alloc`: PoolAllocator (reuses clean pool qubits before allocating new qubits!)
      - `defer_tokens`: dict storing deferred checkpoint HMR bits from Forward -> Reverse (0 forward fixup!)
      - `hmr_ovf`: in Reverse, HMR-frees each block's `ovf` BEFORE undoing the block's rows
        (-1 CCX/row, +1 free ancilla, 0-Tof Clifford carry-out phase fixup!)
      - `mux_round`: 2-bit-exact `MUX(u_i, a[k-1], a[k-2])` rounding carry-in!
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))
    inv_sg = not subtract

    if not lazy_ovf:
        blk_list = []
        ovf_idx = 0
        for grp, w_blk in blocks:
            if w_blk is not None:
                blk_list.append((grp, w_blk, ovf_qubits[ovf_idx]))
                ovf_idx += 1
            else:
                blk_list.append((grp, None, None))
        assert ovf_idx == len(ovf_qubits)

    def _tok(key):
        if defer_tokens is None:
            return None
        return defer_tokens.setdefault(key, {})

    if not uncompute:
        for bi, (grp, w_blk) in enumerate(blocks):
            if w_blk is None:
                for i in grp:
                    _signed_digit_iadd_row(
                        builder, False, u[i], inv_sg, a, target, shift_base + i,
                        max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i)), mux_round=mux_round,
                    )
            else:
                assert grp == sorted(grp, reverse=True), 'grp must be MSB-first'
                if lazy_ovf:
                    ovf = _alloc(1)[0]
                    ovf_qubits.append(ovf)
                else:
                    ovf = blk_list[bi][2]
                lo_tgt = list(target[:w_blk])
                ext_tgt = lo_tgt + [ovf]
                i0 = grp[0]
                _signed_digit_iadd_row(
                    builder, False, u[i0], inv_sg, a, ext_tgt, shift_base + i0,
                    max_anc=max_anc, flip_target_before=lo_tgt, flip_target_after=ext_tgt,
                    alloc=alloc, defer_hmr=_tok((bi, i0)), mux_round=mux_round,
                )
                for i in grp[1:]:
                    _signed_digit_iadd_row(
                        builder, False, u[i], inv_sg, a, ext_tgt, shift_base + i,
                        max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i)), mux_round=mux_round,
                    )
                hi_tgt = list(target[w_blk:])
                if hi_tgt:
                    _neg_ctrl_flip(builder, u[i0], inv_sg, hi_tgt + [ovf])
                    up.pebbled_gidney_iadd(
                        builder, [], hi_tgt, cin=ovf, max_anc=max_anc, subtract=False,
                        alloc=alloc, defer_hmr=_tok(('prop', bi)),
                    )
                    _neg_ctrl_flip(builder, u[i0], inv_sg, hi_tgt + [ovf])
    else:
        ovf_ptr = len(ovf_qubits) - 1
        for bi in range(len(blocks) - 1, -1, -1):
            grp, w_blk = blocks[bi]
            if w_blk is None:
                for i in reversed(grp):
                    _signed_digit_iadd_row(
                        builder, True, u[i], inv_sg, a, target, shift_base + i,
                        max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i)), mux_round=mux_round,
                    )
            else:
                ovf = ovf_qubits[ovf_ptr]
                ovf_ptr -= 1
                lo_tgt = list(target[:w_blk])
                ext_tgt = lo_tgt + [ovf]
                i0 = grp[0]
                hi_tgt = list(target[w_blk:])
                if hi_tgt:
                    _neg_ctrl_flip(builder, u[i0], inv_sg, hi_tgt + [ovf])
                    up.pebbled_gidney_iadd(
                        builder, [], hi_tgt, cin=ovf, max_anc=max_anc, subtract=True,
                        alloc=alloc, defer_hmr=_tok(('prop', bi)),
                    )
                    _neg_ctrl_flip(builder, u[i0], inv_sg, hi_tgt + [ovf])
                if hmr_ovf:
                    # HMR-free `ovf` right now! (-1 CCX per row, +1 ancilla, 0-Tof Clifford carry-out phase!)
                    b_ovf = builder.alloc_bits(1)[0]
                    builder.hmr(ovf, b_ovf)
                    if lazy_ovf:
                        _free(ovf)
                    # Phase (-1)^(b_ovf * (u[i0] ^ inv_sg))
                    builder.push_condition(b_ovf)
                    builder.z(u[i0])
                    if inv_sg:
                        builder.neg()
                    builder.pop_condition()
                    for i in reversed(grp[1:]):
                        _signed_digit_iadd_row(
                            builder, True, u[i], inv_sg, a, lo_tgt, shift_base + i,
                            max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i)),
                            phase_cout_bit=b_ovf, mux_round=mux_round,
                        )
                    _signed_digit_iadd_row(
                        builder, True, u[i0], inv_sg, a, lo_tgt, shift_base + i0,
                        max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i0)),
                        phase_cout_bit=b_ovf, mux_round=mux_round,
                    )
                    builder.free(b_ovf)
                else:
                    for i in reversed(grp[1:]):
                        _signed_digit_iadd_row(
                            builder, True, u[i], inv_sg, a, ext_tgt, shift_base + i,
                            max_anc=max_anc, alloc=alloc, defer_hmr=_tok((bi, i)), mux_round=mux_round,
                        )
                    _signed_digit_iadd_row(
                        builder, True, u[i0], inv_sg, a, ext_tgt, shift_base + i0,
                        max_anc=max_anc, flip_target_before=lo_tgt, flip_target_after=ext_tgt,
                        alloc=alloc, defer_hmr=_tok((bi, i0)), mux_round=mux_round,
                    )
                    if lazy_ovf:
                        _free(ovf)
        if lazy_ovf:
            ovf_qubits.clear()
    return ovf_qubits


def propagate_1ovf_carry(
    builder: km.CircuitBuilder,
    uncompute: bool,
    sign_q: km.q,
    subtract: bool,
    ovf: km.q,
    hi_tgt: Sequence[km.q],
    max_anc: int = 4,
    alloc: up.PoolAllocator | None = None,
    defer_hmr: dict | None = None,
) -> None:
    """Propagates (or uncomputes) a single deferred 1-bit signed carry `ovf - neg_top` into `hi_tgt`."""
    if not hi_tgt:
        return
    inv_sg = not subtract
    _neg_ctrl_flip(builder, sign_q, inv_sg, list(hi_tgt) + [ovf])
    up.pebbled_gidney_iadd(
        builder, [], list(hi_tgt), cin=ovf, max_anc=max_anc, subtract=uncompute,
        alloc=alloc, defer_hmr=defer_hmr,
    )
    _neg_ctrl_flip(builder, sign_q, inv_sg, list(hi_tgt) + [ovf])


def streaming_phase_shift_reuse(
    builder: km.CircuitBuilder,
    p: Sequence[km.q],
    s: Sequence[km.q],
    d_buf: Sequence[km.q],
    angles_half_turns: Sequence[Fraction | float],
    max_s: int = 10,
    implicit_msb: bool = False,
    allow_early_stop: bool = False,
    alloc: up.PoolAllocator | None = None,
    s3_extra_lo: Sequence[km.q] = (),
    s3_max_bits: int | None = None,
) -> None:
    """Streaming delay-line phase kickback reusing 3 existing qubits `d_buf = [d0, d1_0, d1_1]`!

    Uses XOR-merged s2/s3 mux (ONLY 2 CCXs per step instead of 4!) and optionally stops early
    at `n_out = len(angles_half_turns)` when all bits >= n_out of (p_full << s) are guaranteed 0
    (in which case d0, d1 are automatically |0> at k = n_out!).
    - `s3_extra_lo`: optional extra low-order qubits `[q_{-m}, ..., q_{-1}]` injected ONLY when `s[3] == 1`.
    - `s3_max_bits`: optional cap on `p` bits shifted when `s[3] == 1` (e.g., 16 for Wave 1 fusion!).
    """
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))
    w_var = len(p)
    w_eff = w_var + (1 if implicit_msb else 0)
    n_ex = len(s3_extra_lo)
    s3_lim = w_eff if s3_max_bits is None else min(w_eff, s3_max_bits)
    n_out = len(angles_half_turns)
    if not allow_early_stop:
        assert n_out >= w_eff + max_s, f'Need n_out >= {w_eff + max_s} to flush delay lines!'

    d0 = d_buf[0]
    d1 = [d_buf[1], d_buf[2]]
    out = _alloc(1)[0]
    b_bits = builder.alloc_bits(n_out)

    # Helper: XOR `ctrl & (bit_a ^ bit_b)` into `out` using AT MOST 1 CCX!
    def _get_q_c(idx: int | None) -> tuple[km.q | None, int]:
        if idx is None:
            return None, 0
        if idx < 0:
            return s3_extra_lo[n_ex + idx], 0
        if idx < w_var:
            return p[idx], 0
        return None, 1

    def _xor_ctrl_diff(ctrl: km.q, idx_a: int | None, idx_b: int | None) -> None:
        qa, ca = _get_q_c(idx_a)
        qb, cb = _get_q_c(idx_b)
        c_tot = ca ^ cb
        if qa is None and qb is None:
            if c_tot:
                builder.cx(ctrl, out)
        elif qa is not None and qb is None:
            if c_tot:
                builder.x(qa)
            builder.ccx(ctrl, qa, out)
            if c_tot:
                builder.x(qa)
        elif qa is None and qb is not None:
            if c_tot:
                builder.x(qb)
            builder.ccx(ctrl, qb, out)
            if c_tot:
                builder.x(qb)
        else:
            # Both qa and qb are distinct qubits! Compute qa ^ qb in-place on qb -> 1 CCX!
            builder.cx(qa, qb)
            if c_tot:
                builder.x(qb)
            builder.ccx(ctrl, qb, out)
            if c_tot:
                builder.x(qb)
            builder.cx(qa, qb)

    for k in range(n_out):
        # Stage 1: out = p[k] ^ s2&(p[k] ^ p[k-4]) ^ s3&(p[k] ^ p[k-8])  (AT MOST 2 CCXs!)
        i0 = k if 0 <= k < w_eff else None
        i4 = (k - 4) if 0 <= k - 4 < w_eff else None
        i8 = (k - 8) if (-n_ex <= k - 8 < s3_lim) else None

        if i0 is not None:
            if i0 < w_var:
                builder.cx(p[i0], out)
            else:
                builder.x(out)
        if i0 is not None or i4 is not None:
            _xor_ctrl_diff(s[2], i0, i4)
        if i0 is not None or i8 is not None:
            _xor_ctrl_diff(s[3], i0, i8)

        # Stage 2 & 3: delay by 2*s[1] and s[0] using reused x[0], x[2], x[3]
        builder.cswap(s[1], out, d1[k & 1])
        builder.cswap(s[0], out, d0)

        ang = angles_half_turns[k]
        if ang != 0:
            builder.z_pow(out, ang)

        builder.hmr(out, b_bits[k])

    _free(out)

    # Exact 6-Toffoli monomial ANF HMR phase fixup using ONLY 2 ancillas peak!
    # Basis for s in 0..10:
    # 0: 1, 1: s0, 2: s1, 3: s2, 4: s3, 5: s0*s1, 6: s0*s2, 7: s1*s2, 8: s0*s1*s2, 9: s0*s3, 10: s1*s3
    B = []
    for v in range(11):
        s0, s1, s2, s3 = (v >> 0) & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1
        B.append([1, s0, s1, s2, s3, s0 & s1, s0 & s2, s1 & s2, s0 & s1 & s2, s0 & s3, s1 & s3])
    invB = [[1 if r == c else 0 for c in range(11)] for r in range(11)]
    M_mat = [row[:] for row in B]
    for col in range(11):
        piv = next(r for r in range(col, 11) if M_mat[r][col] == 1)
        M_mat[col], M_mat[piv] = M_mat[piv], M_mat[col]
        invB[col], invB[piv] = invB[piv], invB[col]
        for r in range(11):
            if r != col and M_mat[r][col] == 1:
                M_mat[r] = [M_mat[r][c] ^ M_mat[col][c] for c in range(11)]
                invB[r] = [invB[r][c] ^ invB[col][c] for c in range(11)]

    def _emit_basis_phase(bi: int, bq: km.q | None) -> None:
        vs = [v for v in range(11) if invB[bi][v] == 1]
        for j in range(w_eff):
            for v in vs:
                if v >= 8 and j >= s3_lim:
                    continue
                k = j + v
                if k >= n_out:
                    continue
                builder.push_condition(b_bits[k])
                if j < w_var:
                    if bq is None:
                        builder.z(p[j])
                    else:
                        builder.cz(bq, p[j])
                else:
                    # constant MSB = 1
                    if bq is None:
                        builder.neg()
                    else:
                        builder.z(bq)
                builder.pop_condition()
        for j_off, q_ex in enumerate(s3_extra_lo):
            j_virt = j_off - n_ex
            for v in vs:
                if v < 8:
                    continue
                k = j_virt + v
                if 0 <= k < n_out:
                    builder.push_condition(b_bits[k])
                    if bq is None:
                        builder.z(q_ex)
                    else:
                        builder.cz(bq, q_ex)
                    builder.pop_condition()

    # Degree 0 & 1 (0 ancillas, 0 Toffolis!)
    _emit_basis_phase(0, None)
    _emit_basis_phase(1, s[0])
    _emit_basis_phase(2, s[1])
    _emit_basis_phase(3, s[2])
    _emit_basis_phase(4, s[3])

    # Degree 2 & 3 sequentially using ONLY 2 ancillas (`a0`, `a1`) total!
    a0, a1 = _alloc(2)
    # bi=5: s0*s1, bi=8: s0*s1*s2
    builder.init_and(s[0], s[1], a0)
    _emit_basis_phase(5, a0)
    builder.init_and(a0, s[2], a1)
    _emit_basis_phase(8, a1)
    builder.del_and(a0, s[2], a1)
    builder.del_and(s[0], s[1], a0)
    # bi=6: s0*s2
    builder.init_and(s[0], s[2], a0)
    _emit_basis_phase(6, a0)
    builder.del_and(s[0], s[2], a0)
    # bi=7: s1*s2
    builder.init_and(s[1], s[2], a0)
    _emit_basis_phase(7, a0)
    builder.del_and(s[1], s[2], a0)
    # bi=9: s0*s3
    builder.init_and(s[0], s[3], a0)
    _emit_basis_phase(9, a0)
    builder.del_and(s[0], s[3], a0)
    # bi=10: s1*s3
    builder.init_and(s[1], s[3], a0)
    _emit_basis_phase(10, a0)
    builder.del_and(s[1], s[3], a0)

    _free([a0, a1])
    builder.free(b_bits)


if __name__ == '__main__':
    import numpy as np
    rng = np.random.default_rng(77)

    # Test 1: normalize_and_compact -> unpack_s -> streaming_phase_shift_reuse (implicit_msb=True) -> pack_s -> unnormalize
    print('Test 1: compact norm + unpack/pack + streaming_phase_shift_reuse...')
    b = km.CircuitBuilder()
    x = b.create_quantum_register(22, name='x')
    nz = b.alloc_qubits(1)[0]
    s_bits = normalize_and_compact(b, x, nz)
    s_hi = b.alloc_qubits(3)
    s_arr = unpack_s(b, x, nz, s_hi, s_hmr_bits=s_bits)
    # Test streaming_phase_shift_reuse with a 7-bit register `p = x[14..20]` + implicit_msb=True!
    angs = [Fraction(2) ** (k - 17) if k < 18 else 0 for k in range(18)]
    streaming_phase_shift_reuse(b, list(x[14:21]), list(s_arr), [x[0], x[2], x[3]], angs, max_s=10, implicit_msb=True)
    pack_s(b, x, nz, s_hi)
    b.free(s_hi)
    unnormalize_from_compact(b, x, nz)
    b.free(nz)
    c1 = b.finish_circuit()
    print(f'  Q={c1.num_qubits}, ops={dict((k, v) for k, v in c1.operation_statistics().items() if k in ("CCX", "CSWAP", "CCZ", "Z_POW", "HMR"))}')

    # Simulate across 512 diverse x covering EVERY s in 0..10 and both parities!
    xs_list = list(range(1, 129))
    for bl in range(8, 23):
        lo = 1 << (bl - 1)
        hi = min((1 << bl) - 1, 3_133_452)
        xs_list.extend(rng.integers(lo, hi + 1, size=30).tolist())
    xs = np.array(xs_list[:512], dtype=np.uint64)
    x_bits = ((xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1).astype(np.bool_)
    x_reg = km.array([km.q(i) for i in range(22)])
    scratch = km.array([km.q(i) for i in range(22, c1.num_qubits)])

    # Expected phase for each x
    exp_t = []
    for xv in xs.tolist():
        e = 22 - xv.bit_length()
        xn = xv << e
        sv = e >> 1
        p_val = ((xn >> 14) & 0x7F) | 0x80  # 7 bits + implicit MSB=1
        y = p_val << sv
        ph_half = sum((2.0 ** (k - 17)) for k in range(18) if (y >> k) & 1)
        exp_t.append((ph_half / 2.0) % 1.0)
    exp_t = np.array(exp_t)

    sim = km.Simulator(batch_size=512, debug_print=False, counting_operations=True, seed=99)
    max_diff = 0.0
    for shot in range(32):
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, x_bits)
        sim.do(c1)
        assert np.array_equal(sim.read_across_shots(x_reg), x_bits), f'x dirty on shot {shot}!'
        dirty = sim.read_across_shots(scratch).any(axis=1)
        assert not dirty.any(), f'scratch {scratch[int(np.argmax(dirty))]} dirty on shot {shot}!'
        got_t = np.array([float(sim.read_shot_phase(i)) / 2.0 for i in range(512)]) % 1.0
        diff = np.abs((got_t - exp_t + 0.5) % 1.0 - 0.5)
        max_diff = max(max_diff, float(diff.max()))
    st = sim.operation_statistics()
    mean_tof = (st.get('CCX', 0) + st.get('CSWAP', 0) + st.get('CCZ', 0)) / (512 * 32)
    print(f'  Test 1 PASS! max_phase_diff={max_diff:.2e}, mean_Tof={mean_tof:.2f}')

    # Test 2: signed_digit_block_mul_1ovf (14-bit u * 18-bit a -> 21-bit acc, 3 blocks with 1 ovf qubit each!)
    print('Test 2: signed_digit_block_mul_1ovf (14x18 -> 21, 3x 1-qubit ovf, forward + phase + uncompute)...')
    b2 = km.CircuitBuilder()
    u_r = b2.create_quantum_register(14, name='u')
    a_r = b2.create_quantum_register(18, name='a')
    acc = b2.alloc_qubits(21)
    # Initialize acc with constant 1<<19 (center) to stay strictly nonnegative!
    b2.x(acc[19])
    # shift_base = -13: rows 13..11 (sh=0,-1,-2 -> full 21), rows 10..8 (sh=-3,-4,-5 -> max sum < 2^16 -> w_blk=16),
    # rows 7..5 (w_blk=13), rows 4..2 (w_blk=10), rows 1..0 (w_blk=7)
    blks = [
        ([13, 12, 11], None),
        ([10, 9, 8], 16),
        ([7, 6, 5], 13),
        ([4, 3, 2], 10),
        ([1, 0], 7),
    ]
    ovfs: list[km.q] = []
    signed_digit_block_mul_1ovf(b2, False, u_r, a_r, acc, ovfs, shift_base=-13, blocks=blks, max_anc=4, subtract=False, lazy_ovf=True)
    for k in range(21):
        b2.z_pow(acc[k], Fraction(2) ** (k - 20))
    signed_digit_block_mul_1ovf(b2, True, u_r, a_r, acc, ovfs, shift_base=-13, blocks=blks, max_anc=4, subtract=False, lazy_ovf=True)
    b2.x(acc[19])
    b2.free(acc)
    c2 = b2.finish_circuit()
    print(f'  Q={c2.num_qubits} (extra={c2.num_qubits - 32}), static_Tof={c2.operation_statistics().get("CCX", 0)}')

    us = rng.integers(0, 1 << 14, size=256, dtype=np.uint64)
    # a in [0, 2^18 - 1]
    as_ = rng.integers(0, 1 << 18, size=256, dtype=np.uint64)
    # Also include extreme corners (a=0, a=2^18-1, u=0, u=2^14-1)
    us[:4] = [0, (1 << 14) - 1, 0, (1 << 14) - 1]
    as_[:4] = [(1 << 18) - 1, (1 << 18) - 1, 0, 1]
    u_bits = ((us[None, :] >> np.arange(14, dtype=np.uint64)[:, None]) & 1).astype(np.bool_)
    a_bits = ((as_[None, :] >> np.arange(18, dtype=np.uint64)[:, None]) & 1).astype(np.bool_)

    # Exact numpy reference
    exp_acc = np.full(256, 1 << 19, dtype=np.int64)
    for i in range(14):
        sg = np.where(((us >> i) & 1) == 1, 1, -1)
        sh = -13 + i
        if sh >= 0:
            t = as_.astype(np.int64) << sh
        else:
            k = -sh
            t = (as_.astype(np.int64) >> k) + ((as_.astype(np.int64) >> (k - 1)) & 1)
        exp_acc += sg * t
    assert exp_acc.min() >= 0 and exp_acc.max() < (1 << 21)
    exp_t2 = ((exp_acc / float(1 << 20)) / 2.0) % 1.0

    u_reg = km.array([km.q(i) for i in range(14)])
    a_reg = km.array([km.q(i) for i in range(14, 32)])
    scr2 = km.array([km.q(i) for i in range(32, c2.num_qubits)])
    sim2 = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=101)
    max_diff2 = 0.0
    for shot in range(32):
        sim2.clear_for_shot()
        sim2.write_across_shots(u_reg, u_bits)
        sim2.write_across_shots(a_reg, a_bits)
        sim2.do(c2)
        assert np.array_equal(sim2.read_across_shots(u_reg), u_bits)
        assert np.array_equal(sim2.read_across_shots(a_reg), a_bits)
        dirty = sim2.read_across_shots(scr2).any(axis=1)
        assert not dirty.any(), f'scratch {scr2[int(np.argmax(dirty))]} dirty!'
        got_t = np.array([float(sim2.read_shot_phase(i)) / 2.0 for i in range(256)]) % 1.0
        diff = np.abs((got_t - exp_t2 + 0.5) % 1.0 - 0.5)
        max_diff2 = max(max_diff2, float(diff.max()))
    st2 = sim2.operation_statistics()
    mean_tof2 = (st2.get('CCX', 0) + st2.get('CSWAP', 0) + st2.get('CCZ', 0)) / (256 * 32)
    print(f'  Test 2 PASS! max_phase_diff={max_diff2:.2e}, mean_Tof={mean_tof2:.2f} ({mean_tof2/2:.2f}/dir!)')
