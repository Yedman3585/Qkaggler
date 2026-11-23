"""Low-qubit, low-Toffoli quantum circuit primitives for `invsqrt` in kickmix.

Contains:
1. `normalize_to_1_2`: Reflection Gray-code binary barrel shifter normalizing 22-bit `x`
   to `x_norm in [2^21, 2^22)` with 5-bit shift `e` (`e[0]=parity`, `e[1..4]=s in 0..10`),
   and freeing `x[21]` via `X(x[21])` since `x_norm[21] == 1` always.
2. `pebbled_gidney_iadd`: Bounded-ancilla pebbled Gidney in-place adder `target += offset (+ cin)`
   using at most `max_anc` carry qubits (and supporting `len(offset) < len(target)`).
3. `signed_digit_block_mul`: Signed-digit (`+- 1`) multiplication with round-to-nearest
   carry-in and block-carry deferred overflow.
4. `streaming_phase_shift`: Streaming delay-line shifter + 1-ancilla `Z_POW` + `HMR`
   with 6-Toffoli monomial phase fixup (eliminates `y1` register and denormalization).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence
import kickmix as km


class PoolAllocator:
    """Allocates clean |0> scratch qubits from a pool (e.g. HMR-freed `addr` or `c0_lo`)
    BEFORE calling `builder.alloc_qubits()`, returning pool qubits cleanly on `free()`."""

    def __init__(self, builder: km.CircuitBuilder, pool: Sequence[km.q] = ()):
        self.builder = builder
        self.pool_set: set[km.q] = set(pool)
        self.avail: list[km.q] = list(pool)

    def add_pool(self, qs: Sequence[km.q]) -> None:
        for q in qs:
            assert q not in self.pool_set, f'Qubit {q} already in pool!'
            self.pool_set.add(q)
            self.avail.append(q)

    def remove_pool(self, qs: Sequence[km.q]) -> None:
        for q in qs:
            self.avail.remove(q)
            self.pool_set.remove(q)

    def alloc(self, n: int) -> list[km.q]:
        if n <= 0:
            return []
        take = min(n, len(self.avail))
        res = [self.avail.pop() for _ in range(take)]
        if take < n:
            res.extend(self.builder.alloc_qubits(n - take))
        return res

    def free(self, qs: km.q | Sequence[km.q]) -> None:
        if isinstance(qs, km.q):
            qs = [qs]
        to_b = []
        for q in qs:
            if q in self.pool_set:
                self.avail.append(q)
            else:
                to_b.append(q)
        if to_b:
            self.builder.free(km.array(to_b))


# ---------------------------------------------------------------------------
# 1. Normalization to [1, 2) via Reflection Gray-Code Barrel Shifter
# ---------------------------------------------------------------------------

def _c_reverse(builder: km.CircuitBuilder, ctrl: km.q, reg: km.array) -> None:
    """Controlled in-place reversal of `reg` (`len(reg)//2` CSWAPs)."""
    n = len(reg)
    for i in range(n // 2):
        builder.cswap(ctrl, reg[i], reg[n - 1 - i])


def _c_reflect_split(builder: km.CircuitBuilder, ctrl: km.q, x: km.array, a: int) -> None:
    """Controlled T_a = Reverse(0..a-1) * Reverse(a..n-1) = R * left_rotate(a) = right_rotate(a) * R."""
    n = len(x)
    if a > 1:
        _c_reverse(builder, ctrl, x[:a])
    if n - a > 1:
        _c_reverse(builder, ctrl, x[a:])


def _is_all_zero_mcx(builder: km.CircuitBuilder, reg: km.array, target: km.q) -> None:
    builder.x(reg)
    if len(reg) == 1:
        builder.cx(reg[0], target)
    elif len(reg) == 2:
        builder.ccx(reg[0], reg[1], target)
    else:
        builder.multi_controlled_x(reg, target)
    builder.x(reg)


def _gray_rotate_0_15(builder: km.CircuitBuilder, inverse: bool, x: km.array, e_lo: km.array) -> None:
    """Left-rotates (or right-rotates if inverse) 22-bit `x` by `r = e_lo` in `0..15` using 51 CSWAPs.

    Uses T_0, T_1, T_3, T_7, T_15 controlled by Gray code `(e0, e0^e1, e1^e2, e2^e3, e3)`.
    Let `T_a = Rev(0..a-1) * Rev(a..n-1)`. Check: `T_a(i) = (a - 1 - i) mod n`.
    Then `T_b(T_a(i)) = (b - 1 - (a - 1 - i)) mod n = (i + (b - a)) mod n` (left-rotate by `b - a`!).
    """
    # Compute Gray code in place on e_lo[0..3] plus c0 = e_lo[0] before overwrite:
    # We need c4 = e3, c3 = e3^e2, c2 = e2^e1, c1 = e1^e0, c0 = e0.
    # Keep e0 in place, compute c1..c3 in place in reverse order?
    # Wait: if we do CX(e0, e1) -> e1'=e1^e0; CX(e1_orig, e2)... so do top-down:
    # Actually top-down: e3 stays e3 (c4); e2 ^= e3 (c3); ... wait:
    #   If we do bottom-up we destroy e0 before c0!
    #   So do: c0 = e0 (needs 1 extra CX? No: apply T_0 controlled by e0 FIRST before modifying e0,
    #   or apply in order: T_0(e0), then e0 ^= e1 -> T_1(e0), then e1 ^= e2 -> T_3(e1),
    #   then e2 ^= e3 -> T_7(e2), then T_15(e3), and restore e2 ^= e3, e1 ^= e2, e0 ^= e1!).
    # Let's check!
    #   Start: (e0, e1, e2, e3)
    #   1. T_0 controlled by e0            (c0 = e0)
    #   2. CX(e1, e0) -> e0 := e0^e1
    #      T_1 controlled by e0            (c1 = e0^e1)
    #   3. CX(e2, e1) -> e1 := e1^e2
    #      T_3 controlled by e1            (c2 = e1^e2)
    #   4. CX(e3, e2) -> e2 := e2^e3
    #      T_7 controlled by e2            (c3 = e2^e3)
    #   5. T_15 controlled by e3           (c4 = e3)
    #   6. Restore: CX(e3, e2); CX(e2, e1); CX(e1, e0) -> (e0, e1, e2, e3)!
    # ZERO extra qubits, 6 CNOTs, 51 CSWAPs!
    if not inverse:
        _c_reflect_split(builder, e_lo[0], x, 0)
        builder.cx(e_lo[1], e_lo[0])
        _c_reflect_split(builder, e_lo[0], x, 1)
        builder.cx(e_lo[2], e_lo[1])
        _c_reflect_split(builder, e_lo[1], x, 3)
        builder.cx(e_lo[3], e_lo[2])
        _c_reflect_split(builder, e_lo[2], x, 7)
        _c_reflect_split(builder, e_lo[3], x, 15)
        builder.cx(e_lo[3], e_lo[2])
        builder.cx(e_lo[2], e_lo[1])
        builder.cx(e_lo[1], e_lo[0])
    else:
        # Exact reverse of the above (every T_a is self-inverse!)
        builder.cx(e_lo[1], e_lo[0])
        builder.cx(e_lo[2], e_lo[1])
        builder.cx(e_lo[3], e_lo[2])
        _c_reflect_split(builder, e_lo[3], x, 15)
        _c_reflect_split(builder, e_lo[2], x, 7)
        builder.cx(e_lo[3], e_lo[2])
        _c_reflect_split(builder, e_lo[1], x, 3)
        builder.cx(e_lo[2], e_lo[1])
        _c_reflect_split(builder, e_lo[0], x, 1)
        builder.cx(e_lo[1], e_lo[0])
        _c_reflect_split(builder, e_lo[0], x, 0)


def _compute_lzc_5bit(builder: km.CircuitBuilder, inverse: bool, x: km.array, e: km.array) -> None:
    """Computes leading-zero count `e` (5 bits, 0..21) of 22-bit `x >= 1` WITHOUT modifying `x`.

    Uses a 20-bit prefix-AND chain (19 CCX forward, 0 CCX backward via `del_and`? No, to keep
    ancillas tiny we use 19 temp qubits freed immediately, costing ONLY 19 CCX forward and 19
    CCX backward? Wait: 19 temp qubits is only 22+5+19 = 46 qubits, way below peak Q ~ 60!).
    Wait: can we compute `e` with ONLY 19 CCX forward and `del_and`? If we free the 19 prefix
    qubits right after XORing `e`, then `del_and` frees them with **0 CCX**!!
    WAIT A SECOND! YES!!
    In forward:
      1. Compute prefix-zero bits `z[k] = (x[21-k : 22] == 0)` for k = 1..21 using `init_and` (19 CCX!).
      2. XOR the 5 bits of `e` from `z[1..21]` using pure CNOTs (0 CCX!):
         Notice `z` is a thermometer code: `z[1] >= z[2] >= ... >= z[21]`.
         The `b`-th bit of `e = sum(z)` is simply `XOR` over all `k` where `bit b` changes from `k-1` to `k`!
         Specifically:
           `e[0] = z[1] ^ z[2] ^ z[3] ^ ... ^ z[21]` (odd-even transitions!)
           `e[1] = z[2] ^ z[4] ^ z[6] ^ ... ^ z[20]`
           `e[2] = z[4] ^ z[8] ^ z[12] ^ z[16] ^ z[20]`
           `e[3] = z[8] ^ z[16]`
           `e[4] = z[16]`!
      3. Immediately uncompute `z` using `del_and` (**0 CCX!!**) and `free(z)`!
    Total cost: **19 CCX forward, 19 CCX backward = 38 CCX total**, peak Q = 22 + 5 + 20 = 47!
    """
    # Actually z[1] = !x[21] (0 CCX), z[2] = z[1] & !x[20] (1 CCX), ..., z[21] = z[20] & !x[1] (1 CCX)
    # Total: 20 `init_and`s (Wait: we don't even need z[21] if x >= 1? Wait: x=1 has 21 leading zeros!
    # So z[21] = (x[1:22] == 0), which IS needed for x=1 -> e=21! Total = 20 CCX!)
    builder.x(x[1:22])
    z = builder.alloc_qubits(20)  # z[j] holds prefix AND of x[21..20-j] == 0, i.e., z_count[j+2], for j=0..19
    # Wait: let p[1] = x[21] (after X), p[2] = z[0], ..., p[21] = z[19].
    builder.init_and(x[21], x[20], z[0])
    for j in range(1, 20):
        builder.init_and(z[j - 1], x[20 - j], z[j])

    # p_arr[k] for k = 1..21
    p = [None, x[21]] + [z[j] for j in range(20)]
    # Thermometer -> binary XOR formulas:
    # For any k in 0..21, bit b of k is sum_{m=1}^{21} [bit b of m != bit b of m-1] * [k >= m] (mod 2)
    # Because [k >= m] == p[m]!
    for b in range(5):
        for m in range(1, 22):
            if ((m >> b) & 1) != (((m - 1) >> b) & 1):
                builder.cx(p[m], e[b])

    # Uncompute z in reverse using del_and (0 Toffolis!)
    for j in range(19, 0, -1):
        builder.del_and(z[j - 1], x[20 - j], z[j])
    builder.del_and(x[21], x[20], z[0])
    builder.free(z)
    builder.x(x[1:22])


def normalize_to_1_2(builder: km.CircuitBuilder, inverse: bool, x: km.array, e: km.array) -> None:
    """Normalizes 22-bit `x >= 1` in-place to `x_norm = x << e in [2^21, 2^22)` and sets 5-bit `e`.

    Also flips `x[21]` to `|0>` after forward normalization (and back to `|1>` on inverse)
    so `x[21]` can be reused as a clean ancilla qubit!
    Total cost: 20 CCX + 57 CSWAP = 77 Toffolis per direction (154 total)!
    """
    if not inverse:
        _compute_lzc_5bit(builder, False, x, e)
        # Shift by 16 if e[4]==1 (top 16 bits x[6:22] are guaranteed 0 when e[4]==1!)
        builder.cswap(e[4], x[0:6], x[16:22])
        # Shift by e[0..3] in 0..15 using reflection Gray-code barrel shifter (51 CSWAPs!)
        _gray_rotate_0_15(builder, False, x, e[:4])
        # x[21] is now guaranteed |1> -> flip to |0> so it is a free clean ancilla!
        builder.x(x[21])
    else:
        builder.x(x[21])
        _gray_rotate_0_15(builder, True, x, e[:4])
        builder.cswap(e[4], x[0:6], x[16:22])
        _compute_lzc_5bit(builder, True, x, e)


# ---------------------------------------------------------------------------
# 2. Bounded-Ancilla Pebbled Gidney In-Place Adder
# ---------------------------------------------------------------------------

def _gidney_carry_step(builder: km.CircuitBuilder, a: km.q, b: km.q, cin: km.q, cout: km.q) -> None:
    """Computes `cout := maj(a, b, cin)` onto clean `|0>` qubit `cout` (1 CCX), modifying `a := a^cin`, `b := b^cin`."""
    builder.cx(cin, a)
    builder.cx(cin, b)
    builder.init_and(a, b, cout)
    builder.cx(cin, cout)


def _gidney_uncompute_step(builder: km.CircuitBuilder, a: km.q, b: km.q, cin: km.q, cout: km.q, *, write_sum: bool) -> None:
    """Uncomputes `cout` (0 CCX via `del_and`!), restoring `a`, and setting `b := a^b^cin` if `write_sum` else restoring `b`."""
    builder.cx(cin, cout)
    builder.del_and(a, b, cout)
    builder.cx(cin, a)
    if write_sum:
        builder.cx(a, b)  # b was b_orig^cin -> becomes b_orig^cin^a_orig = sum!
    else:
        builder.cx(cin, b)  # restore b_orig


def _gidney_half_step(builder: km.CircuitBuilder, b: km.q, cin: km.q, cout: km.q) -> None:
    """Half-adder carry step when offset bit a == 0: `cout := b & cin` (1 CCX)."""
    builder.init_and(cin, b, cout)


def _gidney_uncompute_half_step(builder: km.CircuitBuilder, b: km.q, cin: km.q, cout: km.q, *, write_sum: bool) -> None:
    builder.del_and(cin, b, cout)
    if write_sum:
        builder.cx(cin, b)


def _gidney_block_forward(
    builder: km.CircuitBuilder,
    a_bits: list[km.q | None],
    b_bits: list[km.q],
    cin: km.q | None,
    carries: list[km.q],
    cout_out: km.q | None,
    *,
    write_sum: bool,
    phase_cout_bit: km.b | None = None,
    internal_z_bits: dict[int, km.b] | None = None,
) -> None:
    """Runs a contiguous Gidney block on `m = len(b_bits)` bits.

    - `carries` has length `m - 1` (clean |0> scratch qubits, returned clean |0>!).
    - If `cout_out is not None`, computes final carry-out onto clean `|0>` qubit `cout_out`.
    - If `cout_out is None` and `phase_cout_bit is not None`, applies `(-1)^cout` conditioned
      on `phase_cout_bit` using 0 Toffolis (pure Cliffords) before writing the top sum bit!
    - If `internal_z_bits` is provided (`{local_carry_pos (1..m): km.b}`), applies 0-Toffoli
      `Z(c_nodes[pos]) if bit` at the peak of the forward sweep (cancelling deferred forward
      HMR checkpoints even when Reverse uses LARGER blocks / unpebbled carries!!).
    - If `write_sum` is True, `b_bits` becomes `b + a + cin (mod 2^m)`.
      If `write_sum` is False, `b_bits` is restored to its original value (used to recompute a carry!).
    Cost: `m` CCX if `cout_out` else `m - 1` CCX.
    """
    m = len(b_bits)
    if m == 0:
        return
    # Carry nodes: c[0] = cin, c[1..m-1] = carries[0..m-2], c[m] = cout_out
    c_nodes: list[km.q | None] = [cin] + list(carries[:max(0, m - 1)]) + [cout_out]

    # Forward sweep: compute carries c[1] .. c[m] (skipping c[m] if cout_out is None)
    last_carry_idx = m if cout_out is not None else m - 1
    for i in range(last_carry_idx):
        ai, bi, ci, co = a_bits[i], b_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is None:
                pass  # carry is 0; shouldn't happen if we only start where inputs exist
            else:
                builder.init_and(ai, bi, co)
        else:
            if ai is None:
                _gidney_half_step(builder, bi, ci, co)
            else:
                _gidney_carry_step(builder, ai, bi, ci, co)

    # Cancel any deferred forward checkpoint HMR bits at internal carry positions (0 Toffolis!)
    if internal_z_bits:
        for pos, mb in internal_z_bits.items():
            cq = c_nodes[pos]
            assert cq is not None, f'internal_z_bits pos={pos} has None carry (m={m}, cout={cout_out})!'
            builder.push_condition(mb)
            builder.z(cq)
            builder.pop_condition()
            builder.free(mb)

    # Top bit (m - 1) optional (-1)^cout phase + sum if cout_out is None
    if cout_out is None:
        i = m - 1
        ai, bi, ci = a_bits[i], b_bits[i], c_nodes[i]
        if phase_cout_bit is not None:
            builder.push_condition(phase_cout_bit)
            if ci is None:
                if ai is not None:
                    builder.cz(ai, bi)
            elif ai is None:
                builder.cz(ci, bi)
            else:
                builder.z(ci)
                builder.cx(ci, ai)
                builder.cx(ci, bi)
                builder.cz(ai, bi)
                builder.cx(ci, bi)
                builder.cx(ci, ai)
            builder.pop_condition()
        if write_sum:
            if ai is not None:
                builder.cx(ai, bi)
            if ci is not None:
                builder.cx(ci, bi)

    # Backward sweep: uncompute c[m-1] .. c[1] (NOT c[m] = cout_out!)
    # Wait: if cout_out is not None, at bit m-1 we computed cout_out from (a_{m-1}^c_{m-1}, b_{m-1}^c_{m-1}),
    # so we must restore a_{m-1} and set/restore b_{m-1} BEFORE uncomputing c_{m-1}!
    if cout_out is not None:
        i = m - 1
        ai, bi, ci = a_bits[i], b_bits[i], c_nodes[i]
        if ci is None:
            # We did init_and(ai, bi, cout_out) without modifying ai or bi.
            # Notice: we do NOT write_sum onto bi until AFTER cout_out is uncomputed?
            # WAIT: if write_sum is True, modifying bi AFTER cout_out is computed is fine for
            # forward computation because cout_out is already computed; ONLY if we later need to
            # uncompute cout_out via del_and(ai, bi, cout_out) would bi need to be restored!
            # We NEVER uncompute cout_out from the modified b_bits directly — we uncompute cout_out
            # using the *inverse* carry recomputation from original/sum bits?
            # NO!! Look carefully:
            # In a pebbled adder, if we keep checkpoint `cout_out` and ALREADY overwrite `b_bits`
            # with `sum`, how do we uncompute `cout_out` later?!
            # CHECK: `cout_out = maj(a, b_orig, cin) = !maj(a, b_sum, cin)`? NO!
            # Wait: `b_sum = a ^ b_orig ^ cin`, so `b_orig = b_sum ^ a ^ cin`!
            # Let's check `maj(a, b_orig, cin)` in terms of `(a, b_sum, cin)`:
            #   `a ^ cin` is `1` iff `a != cin`.
            #   - If `a == cin`: `maj(a, b_orig, cin) = a` (independent of `b_orig`!).
            #   - If `a != cin`: `b_sum = b_orig ^ 1 = !b_orig`, so `maj(a, b_orig, cin) = b_orig = !b_sum`!
            #   **THEREFORE `maj(a, b_orig, cin) == maj(a, !b_sum, cin)` ALWAYS!!**
            #   **CHECK ALL 8 CASES:**
            #     a=0, c=0: b=0 -> sum=0, !sum=1 -> maj(0,1,0)=0 == maj(0,0,0)=0!
            #     a=0, c=0: b=1 -> sum=1, !sum=0 -> maj(0,0,0)=0 == maj(0,1,0)=0!
            #     a=0, c=1: b=0 -> sum=1, !sum=0 -> maj(0,0,1)=0 == maj(0,0,1)=0!
            #     a=0, c=1: b=1 -> sum=0, !sum=1 -> maj(0,1,1)=1 == maj(0,1,1)=1!
            #     a=1, c=0: b=0 -> sum=1, !sum=0 -> maj(1,0,0)=0 == maj(1,0,0)=0!
            #     a=1, c=0: b=1 -> sum=0, !sum=1 -> maj(1,1,0)=1 == maj(1,1,0)=1!
            #     a=1, c=1: b=0 -> sum=0, !sum=1 -> maj(1,1,1)=1 == maj(1,0,1)=1!
            #     a=1, c=1: b=1 -> sum=1, !sum=0 -> maj(1,0,1)=1 == maj(1,1,1)=1!
            #   IT HOLDS FOR ALL 8 CASES! (Because `b_sum` = `b_orig + a + cin`, subtracting `a + cin`
            #   from `b_sum` is literally adding `a + cin` to `~b_sum` and checking carry = `~borrow`!)
            if write_sum and ai is not None:
                builder.cx(ai, bi)
        else:
            if ai is None:
                if write_sum:
                    builder.cx(ci, bi)
            else:
                builder.cx(ci, ai)
                if write_sum:
                    builder.cx(ai, bi)
                else:
                    builder.cx(ci, bi)

    for i in range(m - 2, -1, -1):
        ai, bi, ci, co = a_bits[i], b_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is not None:
                builder.del_and(ai, bi, co)
                if write_sum:
                    builder.cx(ai, bi)
        else:
            if ai is None:
                _gidney_uncompute_half_step(builder, bi, ci, co, write_sum=write_sum)
            else:
                _gidney_uncompute_step(builder, ai, bi, ci, co, write_sum=write_sum)


def _gidney_block_uncompute_cout(
    builder: km.CircuitBuilder,
    a_bits: list[km.q | None],
    b_sum_bits: list[km.q],
    cin: km.q | None,
    carries: list[km.q],
    cout_to_clear: km.q,
) -> None:
    """Clears `cout_to_clear` via HMR + classically-conditioned phase fixup (0.5*(m-1) mean CCX!).

    Applies `HMR(cout_to_clear) -> m_bit`. When `m_bit == 0` (50% of shots), zero Toffolis execute!
    When `m_bit == 1` (50% of shots), recomputes `c[1]..c[m-1]` using `maj(a, !b_sum, cin)`,
    applies the Clifford Z/CZ phase `(-1)^(c_m)`, and uncomputes `c[m-1]..c[1]` via `del_and` (0 CCX).
    """
    m = len(b_sum_bits)
    m_bit = builder.alloc_bits(1)[0]
    builder.hmr(cout_to_clear, m_bit)

    builder.push_condition(m_bit)
    b_arr = km.array(b_sum_bits)
    builder.x(b_arr)
    c_nodes: list[km.q | None] = [cin] + list(carries[:max(0, m - 1)])

    # Forward sweep for c[1]..c[m-1] (only executes when m_bit == 1!)
    for i in range(m - 1):
        ai, bi, ci, co = a_bits[i], b_sum_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is not None:
                builder.init_and(ai, bi, co)
        else:
            if ai is None:
                _gidney_half_step(builder, bi, ci, co)
            else:
                _gidney_carry_step(builder, ai, bi, ci, co)

    # Apply (-1)^(c_m) phase at i = m - 1 using pure Cliffords (0 CCX!)
    i = m - 1
    ai, bi, ci = a_bits[i], b_sum_bits[i], c_nodes[i]
    if ci is None:
        if ai is not None:
            builder.cz(ai, bi)
    else:
        if ai is None:
            builder.cz(ci, bi)
        else:
            # c_m = ci ^ ((ai ^ ci) & (bi ^ ci))
            builder.z(ci)
            builder.cx(ci, ai)
            builder.cx(ci, bi)
            builder.cz(ai, bi)
            builder.cx(ci, bi)
            builder.cx(ci, ai)

    # Uncompute c[m-1]..c[1] (0 CCX!)
    for i in range(m - 2, -1, -1):
        ai, bi, ci, co = a_bits[i], b_sum_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is not None:
                builder.del_and(ai, bi, co)
        else:
            if ai is None:
                _gidney_uncompute_half_step(builder, bi, ci, co, write_sum=False)
            else:
                _gidney_uncompute_step(builder, ai, bi, ci, co, write_sum=False)

    builder.x(b_arr)
    builder.pop_condition()
    builder.free(m_bit)


def _gidney_block_phase_cout(
    builder: km.CircuitBuilder,
    a_bits: list[km.q | None],
    b_bits: list[km.q],
    cin: km.q | None,
    work: list[km.q],
) -> None:
    """Applies (-1)^cout(a_bits, b_bits, cin) without modifying a_bits or b_bits."""
    m = len(b_bits)
    c_nodes: list[km.q | None] = [cin] + work[: m - 1]
    for i in range(m - 1):
        ai, bi, ci, co = a_bits[i], b_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is not None:
                builder.init_and(ai, bi, co)
        else:
            if ai is None:
                _gidney_half_step(builder, bi, ci, co)
            else:
                _gidney_carry_step(builder, ai, bi, ci, co)
    i = m - 1
    ai, bi, ci = a_bits[i], b_bits[i], c_nodes[i]
    if ci is None:
        if ai is not None:
            builder.cz(ai, bi)
    else:
        if ai is None:
            builder.cz(ci, bi)
        else:
            builder.z(ci)
            builder.cx(ci, ai)
            builder.cx(ci, bi)
            builder.cz(ai, bi)
            builder.cx(ci, bi)
            builder.cx(ci, ai)
    for i in range(m - 2, -1, -1):
        ai, bi, ci, co = a_bits[i], b_bits[i], c_nodes[i], c_nodes[i + 1]
        assert co is not None
        if ci is None:
            if ai is not None:
                builder.del_and(ai, bi, co)
        else:
            if ai is None:
                _gidney_uncompute_half_step(builder, bi, ci, co, write_sum=False)
            else:
                _gidney_uncompute_step(builder, ai, bi, ci, co, write_sum=False)


def _pebbled_gidney_phase_cout(
    builder: km.CircuitBuilder,
    a_bits: list[km.q | None],
    b_bits: list[km.q],
    cin: km.q | None,
    max_anc: int | None,
    alloc: PoolAllocator | None = None,
) -> None:
    """Applies (-1)^cout(a_bits, b_bits, cin) using <= max_anc ancillas without modifying inputs."""
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))
    n = len(b_bits)
    if n == 0:
        if cin is not None:
            builder.z(cin)
        return
    if max_anc is None or (n - 1) <= max_anc:
        work = _alloc(max(0, n - 1))
        _gidney_block_phase_cout(builder, a_bits, b_bits, cin, work)
        if work:
            _free(work)
        return
    mid = n // 2
    c_mid = _alloc(1)[0]
    sub_anc = max(1, max_anc - 1)
    _pebbled_gidney_core(builder, a_bits[:mid], b_bits[:mid], cin, c_mid, sub_anc, write_sum=False, alloc=alloc)
    _pebbled_gidney_phase_cout(builder, a_bits[mid:], b_bits[mid:], c_mid, sub_anc, alloc=alloc)
    m_bit = builder.alloc_bits(1)[0]
    builder.hmr(c_mid, m_bit)
    builder.push_condition(m_bit)
    _pebbled_gidney_phase_cout(builder, a_bits[:mid], b_bits[:mid], cin, sub_anc, alloc=alloc)
    builder.pop_condition()
    builder.free(m_bit)
    _free(c_mid)


def _pebbled_gidney_core(
    builder: km.CircuitBuilder,
    a_bits: list[km.q | None],
    b_bits: list[km.q],
    cin: km.q | None,
    cout: km.q | None,
    max_anc: int | None,
    write_sum: bool,
    alloc: PoolAllocator | None = None,
    defer_hmr: dict | None = None,
    phase_cout_bit: km.b | None = None,
) -> None:
    _alloc = (lambda k: alloc.alloc(k)) if alloc is not None else (lambda k: list(builder.alloc_qubits(k)))
    _free = (lambda qs: alloc.free(qs)) if alloc is not None else (lambda qs: builder.free(qs))

    n = len(b_bits)
    if n == 0:
        if cout is not None and cin is not None:
            builder.cx(cin, cout)
        elif phase_cout_bit is not None and cin is not None:
            builder.push_condition(phase_cout_bit)
            builder.z(cin)
            builder.pop_condition()
        return

    # Mode detection for `defer_hmr`:
    # - `defer_record`: empty dict passed in Forward -> HMR checkpoints immediately with 0 fixup!
    # - `defer_replay`: populated dict passed in Reverse -> use SAME checkpoint boundaries and
    #    cancel forward HMR phases with 0-Toffoli `Z(ckpt) if bit`, then uncompute normally!
    defer_record = (defer_hmr is not None and 'bits' not in defer_hmr)
    defer_replay = (defer_hmr is not None and 'bits' in defer_hmr)

    if defer_replay:
        forced_blocks = defer_hmr['blocks']
        fwd_bits = defer_hmr['bits']
        if forced_blocks == 'no_defer':
            defer_replay = False
            defer_hmr = None
        elif forced_blocks is None:
            # Forward was unpebbled (0 checkpoints)!
            carries = _alloc(max(0, n - 1))
            _gidney_block_forward(
                builder, a_bits, b_bits, cin, carries, cout,
                write_sum=write_sum, phase_cout_bit=phase_cout_bit,
            )
            if carries:
                _free(carries)
            return
        else:
            # Map each forward checkpoint carry position `hi_k` -> `fwd_bits[k]`!
            fwd_ckpt_map: dict[int, km.b] = {
                forced_blocks[k][1]: fwd_bits[k] for k in range(len(fwd_bits))
            }
            # ASYMMETRIC ZERO-OVERHEAD REPLAY: compute optimal blocks for CURRENT `max_anc`!
            # Case 1: Current `max_anc` fits unpebbled (`n - 1 <= max_anc`) -> ZERO conditional CCXs!!
            if max_anc is None or (n - 1) <= max_anc:
                # All forward checkpoints `hi_k < n` sit right on `carries[hi_k - 1]`!
                assert all(1 <= p < n for p in fwd_ckpt_map), f'Invalid fwd_ckpt_map {fwd_ckpt_map} for n={n}'
                carries = _alloc(max(0, n - 1))
                _gidney_block_forward(
                    builder, a_bits, b_bits, cin, carries, cout,
                    write_sum=write_sum, phase_cout_bit=phase_cout_bit,
                    internal_z_bits=fwd_ckpt_map,
                )
                if carries:
                    _free(carries)
                return

            # Case 2: Build optimal staircase blocks for CURRENT `max_anc` (often larger than A_fwd!)
            # If current `max_anc` fits a better (fewer-checkpoint) staircase, use it and cancel
            # any forward checkpoint inside the corresponding reverse block's internal carries!
            rev_blocks = None
            if max_anc >= 2:
                A = max_anc
                cand_blocks: list[tuple[int, int]] = []
                pos = 0
                k = 0
                fits = False
                while pos < n:
                    rem = n - pos
                    if k + (rem - 1) <= A:
                        cand_blocks.append((pos, n))
                        fits = True
                        break
                    L = A - k
                    if L < 2:
                        break
                    cand_blocks.append((pos, pos + L))
                    pos += L
                    k += 1
                if fits:
                    rev_blocks = cand_blocks

            if rev_blocks is None:
                # Find smallest A_min >= max_anc that fits a 1-level staircase (avoids forced_blocks Q spike!)
                A_min = max(2, max_anc)
                while True:
                    cand_blocks = []
                    pos = 0
                    k = 0
                    fits = False
                    while pos < n:
                        rem = n - pos
                        if k + (rem - 1) <= A_min:
                            cand_blocks.append((pos, n))
                            fits = True
                            break
                        L = A_min - k
                        if L < 2:
                            break
                        cand_blocks.append((pos, pos + L))
                        pos += L
                        k += 1
                    if fits:
                        rev_blocks = cand_blocks
                        break
                    A_min += 1

            assert rev_blocks[-1][0] < n, f'Invalid rev_blocks {rev_blocks} for n={n}'
            n_ckpts = len(rev_blocks) - 1
            ckpts: list[km.q] = []
            for bi_idx, (lo, hi) in enumerate(rev_blocks):
                L = hi - lo
                blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
                blk_cout = _alloc(1)[0] if bi_idx < n_ckpts else cout
                if bi_idx < n_ckpts:
                    ckpts.append(blk_cout)
                # Collect all forward checkpoints falling in (lo, hi] -> local carry index (p - lo) in 1..L!
                local_z = {p - lo: mb for p, mb in fwd_ckpt_map.items() if lo < p <= hi}
                work = _alloc(max(0, L - 1))
                _gidney_block_forward(
                    builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, blk_cout,
                    write_sum=write_sum,
                    phase_cout_bit=(phase_cout_bit if bi_idx == n_ckpts else None),
                    internal_z_bits=(local_z if local_z else None),
                )
                if work:
                    _free(work)
            for bi_idx in range(n_ckpts - 1, -1, -1):
                lo, hi = rev_blocks[bi_idx]
                L = hi - lo
                blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
                work = _alloc(max(0, L - 1))
                if write_sum:
                    _gidney_block_uncompute_cout(
                        builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, ckpts[bi_idx]
                    )
                else:
                    m_bit = builder.alloc_bits(1)[0]
                    builder.hmr(ckpts[bi_idx], m_bit)
                    builder.push_condition(m_bit)
                    _gidney_block_phase_cout(builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work)
                    builder.pop_condition()
                    builder.free(m_bit)
                if work:
                    _free(work)
                _free(ckpts[bi_idx])
            return

    need_direct = n - 1
    if max_anc is None or need_direct <= max_anc:
        if defer_record:
            defer_hmr['blocks'] = None
            defer_hmr['bits'] = []
        carries = _alloc(max(0, n - 1))
        _gidney_block_forward(
            builder, a_bits, b_bits, cin, carries, cout,
            write_sum=write_sum, phase_cout_bit=phase_cout_bit,
        )
        if carries:
            _free(carries)
        return

    # 1-level STAIRCASE pebbling (OR arbitrary small-ancilla chain when defer_record=True!!):
    # Wait: when `defer_record=True`, Forward only ever keeps 2 checkpoints alive at once
    # (`prev_ckpt` + `blk_cout` + `work`), so ANY `max_anc >= 2` can partition ANY `n` into
    # blocks of size `L = max_anc - 1` (or `max_anc` for first block) with ZERO recursion and
    # ZERO conditional CCXs!!
    if max_anc >= 2:
        A = max_anc
        st_blocks = []
        if defer_record:
            # Forward deferred-HMR chain: block 0 has size A (no prev_ckpt!), subsequent blocks have
            # size max(1, A - 1) (1 prev_ckpt + 1 blk_cout + (L-1) work <= A => L <= A - 1), and
            # last block has size <= A (1 prev_ckpt + (L-1) work <= A => L <= A)!
            # Even at A=2: blocks [2, 1, 1, ..., 2] fit ANY n with ONLY 2 ancillas & 0 cond CCXs!
            pos = 0
            while pos < n:
                rem = n - pos
                # Last block (no blk_cout): needs (1 if pos > 0 else 0) + (rem - 1) <= A
                max_last = A if pos > 0 else (A + 1)
                if rem <= max_last:
                    st_blocks.append((pos, n))
                    break
                # Intermediate block (needs blk_cout):
                # pos == 0: 1 (blk_cout) + (L - 1) work <= A => L = A
                # pos > 0 : 1 (prev_ckpt) + 1 (blk_cout) + (L - 1) work <= A => L = max(1, A - 1)
                L = A if pos == 0 else max(1, A - 1)
                # Ensure last block has at least 2 bits (or >= 1 bit) so hi_k < n - 1 when possible!
                if pos + L >= n - 1 and n - pos >= 3:
                    L = max(1, n - pos - 2)
                st_blocks.append((pos, pos + L))
                pos += L
            fits = True
        else:
            pos = 0
            k = 0
            fits = False
            while pos < n:
                rem = n - pos
                if k + (rem - 1) <= A:
                    st_blocks.append((pos, n))
                    fits = True
                    break
                L = A - k
                if L < 2:
                    break
                st_blocks.append((pos, pos + L))
                pos += L
                k += 1

        if fits:
            n_ckpts = len(st_blocks) - 1
            if defer_record:
                # Forward deferred HMR: only 2 checkpoints live at once, ZERO conditional CCXs!
                hmr_bits: list[km.b] = []
                prev_ckpt: km.q | None = None
                for bi_idx, (lo, hi) in enumerate(st_blocks):
                    L = hi - lo
                    blk_cin = cin if bi_idx == 0 else prev_ckpt
                    blk_cout = _alloc(1)[0] if bi_idx < n_ckpts else cout
                    work = _alloc(max(0, L - 1))
                    _gidney_block_forward(
                        builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, blk_cout,
                        write_sum=write_sum,
                        phase_cout_bit=(phase_cout_bit if bi_idx == n_ckpts else None),
                    )
                    if work:
                        _free(work)
                    if prev_ckpt is not None:
                        mb = builder.alloc_bits(1)[0]
                        builder.hmr(prev_ckpt, mb)
                        _free(prev_ckpt)
                        hmr_bits.append(mb)
                    prev_ckpt = blk_cout
                assert prev_ckpt is cout and len(hmr_bits) == n_ckpts
                defer_hmr['blocks'] = st_blocks
                defer_hmr['bits'] = hmr_bits
                return

            ckpts = []
            for bi_idx, (lo, hi) in enumerate(st_blocks):
                L = hi - lo
                blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
                blk_cout = _alloc(1)[0] if bi_idx < n_ckpts else cout
                if bi_idx < n_ckpts:
                    ckpts.append(blk_cout)
                work = _alloc(max(0, L - 1))
                _gidney_block_forward(
                    builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, blk_cout,
                    write_sum=write_sum,
                    phase_cout_bit=(phase_cout_bit if bi_idx == n_ckpts else None),
                )
                if work:
                    _free(work)
            for bi_idx in range(n_ckpts - 1, -1, -1):
                lo, hi = st_blocks[bi_idx]
                L = hi - lo
                blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
                work = _alloc(max(0, L - 1))
                if write_sum:
                    _gidney_block_uncompute_cout(
                        builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, ckpts[bi_idx]
                    )
                else:
                    m_bit = builder.alloc_bits(1)[0]
                    builder.hmr(ckpts[bi_idx], m_bit)
                    builder.push_condition(m_bit)
                    _gidney_block_phase_cout(builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work)
                    builder.pop_condition()
                    builder.free(m_bit)
                if work:
                    _free(work)
                _free(ckpts[bi_idx])
            return

    if defer_record:
        defer_hmr['blocks'] = 'no_defer'
        defer_hmr['bits'] = []

    # 2-level recursive pebbling: split at mid using 1 super-checkpoint c_mid!
    if max_anc >= 3 and n >= 6:
        mid = n // 2
        c_mid = _alloc(1)[0]
        sub_anc = max_anc - 1
        _pebbled_gidney_core(builder, a_bits[:mid], b_bits[:mid], cin, c_mid, sub_anc, write_sum=write_sum, alloc=alloc)
        _pebbled_gidney_core(
            builder, a_bits[mid:], b_bits[mid:], c_mid, cout, sub_anc,
            write_sum=write_sum, alloc=alloc, phase_cout_bit=phase_cout_bit,
        )
        m_bit = builder.alloc_bits(1)[0]
        builder.hmr(c_mid, m_bit)
        builder.push_condition(m_bit)
        if write_sum:
            builder.x(b_bits[:mid])
        _pebbled_gidney_phase_cout(builder, a_bits[:mid], b_bits[:mid], cin, sub_anc, alloc=alloc)
        if write_sum:
            builder.x(b_bits[:mid])
        builder.pop_condition()
        builder.free(m_bit)
        _free(c_mid)
        return

    # Fallback: minimal 1-level B
    best_B = max(2, int(n ** 0.5))
    blocks = [(s, min(s + best_B, n)) for s in range(0, n, best_B)]
    n_ckpts = len(blocks) - 1
    ckpts = _alloc(n_ckpts)
    work = _alloc(best_B - 1)
    for bi_idx, (lo, hi) in enumerate(blocks):
        blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
        blk_cout = ckpts[bi_idx] if bi_idx < n_ckpts else cout
        _gidney_block_forward(
            builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, blk_cout,
            write_sum=write_sum,
            phase_cout_bit=(phase_cout_bit if bi_idx == n_ckpts else None),
        )
    for bi_idx in range(n_ckpts - 1, -1, -1):
        lo, hi = blocks[bi_idx]
        blk_cin = cin if bi_idx == 0 else ckpts[bi_idx - 1]
        if write_sum:
            _gidney_block_uncompute_cout(
                builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work, ckpts[bi_idx]
            )
        else:
            m_bit = builder.alloc_bits(1)[0]
            builder.hmr(ckpts[bi_idx], m_bit)
            builder.push_condition(m_bit)
            _gidney_block_phase_cout(builder, a_bits[lo:hi], b_bits[lo:hi], blk_cin, work)
            builder.pop_condition()
            builder.free(m_bit)
    _free(work)
    _free(ckpts)


def pebbled_gidney_iadd(
    builder: km.CircuitBuilder,
    offset: km.array,
    target: km.array,
    *,
    cin: km.q | None = None,
    max_anc: int | None = None,
    subtract: bool = False,
    alloc: PoolAllocator | None = None,
    defer_hmr: dict | None = None,
    phase_cout_bit: km.b | None = None,
) -> None:
    """In-place `target += offset (+ cin)` (or `-=` if `subtract`) mod 2^len(target).

    Supports PoolAllocator, deferred checkpoint HMR (`defer_hmr`), and 0-Tof carry-out phase
    fixup (`phase_cout_bit`) for `HMR(ovf)` in reverse block multiplication!
    """
    n = len(target)
    if n == 0:
        return
    if subtract:
        builder.x(target)
        pebbled_gidney_iadd(
            builder, offset, target, cin=cin, max_anc=max_anc, subtract=False,
            alloc=alloc, defer_hmr=defer_hmr, phase_cout_bit=phase_cout_bit,
        )
        builder.x(target)
        return
    if len(offset) == 0 and cin is None:
        if defer_hmr is not None and 'bits' not in defer_hmr:
            defer_hmr['blocks'] = None
            defer_hmr['bits'] = []
        return
    a_bits: list[km.q | None] = [offset[i] if i < len(offset) else None for i in range(n)]
    b_bits: list[km.q] = list(target)
    _pebbled_gidney_core(
        builder, a_bits, b_bits, cin, None, max_anc, write_sum=True,
        alloc=alloc, defer_hmr=defer_hmr, phase_cout_bit=phase_cout_bit,
    )


# ---------------------------------------------------------------------------
# 3. Signed-Digit (+-1) Addition & Block-Carry Multiplier
# ---------------------------------------------------------------------------

def signed_digit_iadd(
    builder: km.CircuitBuilder,
    sign_bit: km.q,
    offset: km.array,
    target: km.array,
    *,
    cin: km.q | None = None,
    max_anc: int | None = None,
    negate_sign: bool = False,
) -> None:
    """Performs `if sign_bit == 1: target += offset (+ cin) else: target -= offset (+ cin)` (or flipped if negate_sign).

    Uses exact 1's-complement flipping: `~(~target + offset + cin) == target - offset - cin`!
    Wait: check `cin` under 1's-complement!
      - If we want `target - (offset + cin)`:
        `~target = 2^W - 1 - target`
        `~target + offset + cin = 2^W - 1 - target + offset + cin`
        `~(...) = target - offset - cin`! EXACT!
    So `CX(ctrl, target)` before & after `pebbled_gidney_iadd` works identically WITH `cin`!
    """
    # When negate_sign=False: sign_bit==1 -> add, sign_bit==0 -> sub (so flip target when sign_bit==0!)
    # When negate_sign=True:  sign_bit==1 -> sub, sign_bit==0 -> add (so flip target when sign_bit==1!)
    if not negate_sign:
        builder.x(sign_bit)
    for q in target:
        builder.cx(sign_bit, q)
    pebbled_gidney_iadd(builder, offset, target, cin=cin, max_anc=max_anc, subtract=False)
    for q in target:
        builder.cx(sign_bit, q)
    if not negate_sign:
        builder.x(sign_bit)


def signed_digit_block_mul(
    builder: km.CircuitBuilder,
    inverse: bool,
    u: km.array,
    a: km.array,
    target: km.array,
    ovf_regs: list[km.array],
    *,
    shift_base: int,
    blocks: list[tuple[list[int], int | None]],
    max_anc: int | None = None,
    subtract: bool = False,
    round_cin: bool = True,
) -> None:
    """Computes `target +=/-= sum_{i} (2*u[i] - 1) * round(a * 2^(shift_base + i))` using Block-Carry!

    `blocks` is a list of `(row_indices, w_blk)` tuples:
    - If `w_blk is None`, rows add directly into full `target` (no overflow qubits).
    - If `w_blk is int`, uses a 2-qubit `ovf` register from `ovf_regs` (one per non-None block),
      adds rows into `target[:w_blk] + ovf`, and propagates `(ovf - 1) in {-1, 0, +1}` into
      `target[w_blk:]` ONCE at the end via a single `signed_digit_iadd` incrementer!
    Note: `ovf_regs` MUST be kept alive between `inverse=False` and `inverse=True`!
    """
    neg_sign = not subtract if inverse else subtract
    w_a = len(a)
    W = len(target)

    def _run_row(i: int, tgt_slice: km.array) -> None:
        sh = shift_base + i
        if sh >= 0:
            # a << sh added into tgt_slice[sh:]
            if sh >= len(tgt_slice):
                return
            off = a[:min(w_a, len(tgt_slice) - sh)]
            signed_digit_iadd(builder, u[i], off, tgt_slice[sh:], cin=None, max_anc=max_anc, negate_sign=neg_sign)
        else:
            k = -sh
            if k > w_a:
                return
            off = a[k:] if k < w_a else km.array([])
            cin_q = a[k - 1] if (round_cin and k - 1 < w_a) else None
            if len(off) == 0 and cin_q is None:
                return
            signed_digit_iadd(builder, u[i], off, tgt_slice, cin=cin_q, max_anc=max_anc, negate_sign=neg_sign)

    # Map each non-None block to its 2-qubit ovf register
    ovf_blocks = []
    oi = 0
    for rows, w_blk in blocks:
        if w_blk is not None:
            ovf_blocks.append((rows, w_blk, ovf_regs[oi]))
            oi += 1
        else:
            ovf_blocks.append((rows, None, None))
    assert oi == len(ovf_regs)

    empty = km.array([])
    if not inverse:
        # 1. Run each block's rows
        for rows, w_blk, ovf in ovf_blocks:
            if w_blk is None:
                for i in rows:
                    _run_row(i, target)
            else:
                builder.x(ovf[0])  # ovf := 1 (represents 0 carry)
                sub_tgt = km.array(list(target[:w_blk]) + [ovf[0], ovf[1]])
                for i in rows:
                    _run_row(i, sub_tgt)
        # 2. Propagate each ovf (ovf - 1 in {-1, 0, +1}) into target[w_blk:]
        # Note: propagate in reverse order of w_blk (or any order — addition commutes!)
        for _, w_blk, ovf in ovf_blocks:
            if w_blk is not None and w_blk < W:
                builder.x(ovf[0])  # now ovf[0] == 1 iff ovf != 1 (nonzero carry!)
                signed_digit_iadd(
                    builder, ovf[1], empty, target[w_blk:],
                    cin=ovf[0], max_anc=max_anc, negate_sign=False,
                )
                builder.x(ovf[0])
    else:
        # Exact reverse of forward!
        for _, w_blk, ovf in reversed(ovf_blocks):
            if w_blk is not None and w_blk < W:
                builder.x(ovf[0])
                signed_digit_iadd(
                    builder, ovf[1], empty, target[w_blk:],
                    cin=ovf[0], max_anc=max_anc, negate_sign=True,
                )
                builder.x(ovf[0])
        for rows, w_blk, ovf in reversed(ovf_blocks):
            if w_blk is None:
                for i in reversed(rows):
                    _run_row(i, target)
            else:
                sub_tgt = km.array(list(target[:w_blk]) + [ovf[0], ovf[1]])
                for i in reversed(rows):
                    _run_row(i, sub_tgt)
                builder.x(ovf[0])


# ---------------------------------------------------------------------------
# 4. Streaming Delay-Line Phasing (Zero y1 register, 6-Toffoli HMR fixup!)
# ---------------------------------------------------------------------------

def streaming_phase_shift(
    builder: km.CircuitBuilder,
    p: km.array,
    s: km.array,
    angles_half_turns: list[float | Fraction],
    *,
    max_s: int = 10,
) -> None:
    """Applies `exp(i * pi * sum_k angles_half_turns[k] * ((p << s) >> k & 1))` for `s in 0..max_s`.

    Uses ONLY 5 ancillas during streaming (`a, d0, d1_0, d1_1, h0`) and 6 monomial ancillas
    briefly for the 6-Toffoli HMR phase fixup (total <= 6 extra qubits!).
    `len(angles_half_turns)` must be `<= len(p) + max_s`.
    """
    w = len(p)
    n_out = len(angles_half_turns)
    # s = s[0] + 2*s[1] + 4*s[2] + 8*s[3], with s <= 10 => s[2] & s[3] == 0 ALWAYS!
    a = builder.alloc_qubits(1)[0]
    d0 = builder.alloc_qubits(1)[0]
    d1 = builder.alloc_qubits(2)
    h0 = builder.alloc_qubits(1)[0]  # h0 = !(s[2] | s[3]) = 1 ^ s[2] ^ s[3]
    builder.x(h0)
    builder.cx(s[2], h0)
    builder.cx(s[3], h0)

    b_bits = builder.alloc_bits(n_out)

    for k in range(n_out):
        # 1. Load a := p[k - 4*s_hi] where 4*s_hi in {0, 4, 8}
        i0, i1, i2 = k, k - 4, k - 8
        in0 = 0 <= i0 < w
        in1 = 0 <= i1 < w
        in2 = 0 <= i2 < w
        if in0 and in1 and in2:
            builder.cx(p[i0], a)
            builder.cx(p[i0], p[i1])
            builder.ccx(s[2], p[i1], a)
            builder.cx(p[i0], p[i1])
            builder.cx(p[i0], p[i2])
            builder.ccx(s[3], p[i2], a)
            builder.cx(p[i0], p[i2])
        else:
            if in0:
                builder.ccx(h0, p[i0], a)
            if in1:
                builder.ccx(s[2], p[i1], a)
            if in2:
                builder.ccx(s[3], p[i2], a)

        # 2. Delay by 2*s[1] + s[0] using d1[k % 2] and d0
        # Wait: at the end of the loop (k == n_out - 1), d1 and d0 must be clean |0>!
        # Note: p[w-1] enters at k = w - 1 + 4*s_hi <= w + 7 (for s_hi = 2 -> s in 8..10).
        # For s = 10 (s3=1, s1=1, s0=0), it exits d1 at k = w + 9.
        # For s = 9  (s3=1, s1=0, s0=1), it exits d0 at k = w + 8.
        # For s = 7  (s2=1, s1=1, s0=1), it enters at w + 3, exits d1 at w + 5, exits d0 at w + 6.
        # So after k = w + 9, d0, d1[0], d1[1] are ALL guaranteed |0>!
        if k < w + 10:
            builder.cswap(s[1], a, d1[k % 2])
        if k < w + 9:
            builder.cswap(s[0], a, d0)

        # 3. Apply Z_POW and HMR -> b_bits[k]
        ang = angles_half_turns[k]
        if ang != 0:
            builder.z_pow(a, ang)
        builder.hmr(a, b_bits[k])

    # Flush d0, d1 if n_out < w + 10 (not needed if n_out == w + 10; if n_out < w + 10,
    # caller must ensure bits k >= n_out are never nonzero!)
    builder.cx(s[3], h0)
    builder.cx(s[2], h0)
    builder.x(h0)
    builder.free(h0)
    builder.free(d1)
    builder.free(d0)
    builder.free(a)

    # 4. Undo ALL HMR phase errors: (-1)^(sum_{j=0}^{w-1} p[j] * b_bits[j + s])
    # Compute the 6 nonlinear monomials of (s0, s1, s2, s3) for s in 0..10 (ONLY 6 Toffolis!):
    # Valid s in 0..10: s2 & s3 == 0, and s0 & s1 & s3 == 0.
    # Monomial basis M = [1, s0, s1, s2, s3, m01, m02, m12, m012, m03, m13] (11 basis functions for 11 states!)
    m = builder.alloc_qubits(6)  # m01, m02, m12, m012, m03, m13
    builder.init_and(s[0], s[1], m[0])      # m01
    builder.init_and(s[0], s[2], m[1])      # m02
    builder.init_and(s[1], s[2], m[2])      # m12
    builder.init_and(m[0], s[2], m[3])      # m012
    builder.init_and(s[0], s[3], m[4])      # m03
    builder.init_and(s[1], s[3], m[5])      # m13

    # Precompute ANF (Mobius transform over s in 0..10) of indicator 1[s == v] in the 11-element basis!
    # Basis qubits: None (const 1), s0, s1, s2, s3, m01, m02, m12, m012, m03, m13
    basis_q: list[km.q | None] = [None, s[0], s[1], s[2], s[3], m[0], m[1], m[2], m[3], m[4], m[5]]
    # Evaluate each basis function at v = 0..10 -> 11x11 binary matrix B, invert over GF(2)!
    B = []
    for v in range(11):
        b0, b1, b2, b3 = v & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1
        B.append([1, b0, b1, b2, b3, b0 & b1, b0 & b2, b1 & b2, b0 & b1 & b2, b0 & b3, b1 & b3])
    # Invert B over GF(2): row `bi` of invB gives which `v` contribute to basis function `bi`!
    invB = [[1 if i == j else 0 for j in range(11)] for i in range(11)]
    M_mat = [row[:] for row in B]
    for col in range(11):
        piv = next(r for r in range(col, 11) if M_mat[r][col] == 1)
        M_mat[col], M_mat[piv] = M_mat[piv], M_mat[col]
        invB[col], invB[piv] = invB[piv], invB[col]
        for r in range(11):
            if r != col and M_mat[r][col] == 1:
                M_mat[r] = [M_mat[r][c] ^ M_mat[col][c] for c in range(11)]
                invB[r] = [invB[r][c] ^ invB[col][c] for c in range(11)]

    # For each bit j of p and each basis function bi:
    # Parity bit = XOR over {v in 0..10 : invB[bi][v] == 1 and j + v < n_out} of b_bits[j + v]
    # Note: we can apply Z/CZ directly per b_bits[k] (just Clifford!) without allocating extra classical bits!
    for j in range(w):
        for bi in range(11):
            bq = basis_q[bi]
            vs = [v for v in range(11) if invB[bi][v] == 1 and j + v < n_out]
            for v in vs:
                k = j + v
                builder.push_condition(b_bits[k])
                if bq is None:
                    builder.z(p[j])
                else:
                    builder.cz(bq, p[j])
                builder.pop_condition()

    # Uncompute the 6 monomials via del_and (0 Toffolis!)
    builder.del_and(s[1], s[3], m[5])
    builder.del_and(s[0], s[3], m[4])
    builder.del_and(m[0], s[2], m[3])
    builder.del_and(s[1], s[2], m[2])
    builder.del_and(s[0], s[2], m[1])
    builder.del_and(s[0], s[1], m[0])
    builder.free(m)
    builder.free(b_bits)
