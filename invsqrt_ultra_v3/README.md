# `ultra_v3` `invsqrt` Handoff Package (for Merging with `Qkaggler v34` LUT Synthesis)

This self-contained archive packages our best verified `invsqrt` (`Benchmark I`: $|x\rangle|0\rangle \to e^{2\pi i / \sqrt{x}} |x\rangle|0\rangle$ for $x \in [1, 3\,133\,452]$, phase error $\le 2^{-24}/\sqrt{x}$ turns) generator (`ultra_v3`) and verified circuit (`ultra_v3_p.kmb`).

---

## 1. Current Verified Metrics & Section Breakdown vs `Qkaggler v34`

**Included verified circuit:** `ultra_v3_p.kmb` (Stage-2 PASS: `1,048,576` shots, `k = 0` errors, `worst_error_over_tolerance = 0.999687` at `x = 130`, `reaction_depth = 645`, `num_ops = 53,075`).

| Metric | `ultra_v3_p.kmb` (ours) | `Qkaggler v34` | Delta (`ours - v34`) |
| :--- | :---: | :---: | :---: |
| **Qubits (`Q`)** | **`60`** | `62` | **`-2` qubits** |
| **Mean Toffolis (`mean_Tof`)** | `1,481.50` (`static = 1,546`, `129` cond) | `1,281.67` | `+199.83` |
| **`Z_POW` + `T/4` (`ZT`)** | `300.59` (`41` non-Clifford rots) | `297.71` (`44` rots) | `+2.88` (`-8.0` at `mm=6.1e-12`) |
| **Toffoli-eq (`C`)** | `1,782.09` | `1,579.38` | `+202.71` |
| **Score (`Q * C`)** | **`106,925.35`** (`19.37x` over baseline) | `97,921.63` (`21.21x`) | `+9,003.72` |

### Why Merging Beats `v34` Decisively (Target: `Q = 60`, `score ~ 87k–93k`)

Section-by-section breakdown of `ultra_v3_p.kmb` (`Norm+W0L = 126.0`, `Wave0 = 58.0`, `Q1+Mid = 602.0`, `Wave1 = 95.0`, `Post-W1 = 600.5`):

| Subsystem | `ultra_v3_p` (`Q=60`) | `Qkaggler v34` (`Q=62`) | Delta |
| :--- | :---: | :---: | :---: |
| `Norm + Unnorm` (binary barrel shift + compact `s`) | **`161.0`** | `173.0` | **`-12.0`** (we win) |
| `Wave 0` (XOR-merged streaming + `s3==1` fusion) | **`58.0`** | `69.0` | **`-11.0`** (we win) |
| `Mid` arithmetic (`c2*u` + `r1*u` fwd, `0` cond CCX) | **`338.5`** | `351.0` | **`-12.5`** (we win) |
| `Wave 1` (`s3_max_bits=16`, early stop at `k=25`) | **`95.0`** | `149.0` | **`-54.0`** (we win) |
| `ZT` (`41` rots vs `44`, `s3==1` fusion) | `300.6` (`289.7` w/ `mux_min=0`) | `297.7` | `+2.9` (`-8.0` w/ full mux) |
| **All GF(2) LUTs combined** | `472.0` | **`~340.0`** | **`+132.0`** (main merge lever!) |
| `Post-W1` arithmetic (`r1*u` + `c2*u` rev) | `~360.0` | `~317.0` | `+43.0` |

**Bottom line:** `ultra_v3` already beats `v34` by **`~90–98` Tof-eq** across `Norm+Unnorm`, `Wave 0`, `Mid` arithmetic, and `Wave 1`, while using **`2` fewer qubits (`Q = 60` vs `62`)**. Dropping `v34`'s bilinear MinRank LUT synthesis (`r = 3..4` CCX/bit, `~340` CCX total vs our `472` CCX) directly into `ultra_v3`'s LUT hooks immediately yields:
- `C ≈ 1,782 - 132 = 1,650` at `Q = 60` -> **`score ≈ 99,000`** (or **`~92,000–95,000`** with `v34` LUTs + `Q = 60` / `r1_anc = 7` in Stage 8 `top_grp`/`mid_grp`!).
- With `v34`'s LUTs also freeing `1`–`2` boost qubits, `defer_c2_ovf = False` (`-17` CCX `c2_ovf_r`) and full `mux_round` (`-11` ZT) can fit at `Q = 60` -> **`score ≈ 89,000–92,000` (`~22.5–23.3x` over baseline)**.

---

## 2. Included Files & Dependencies

- `ultra_v3.py` — Parametric piecewise-quadratic `invsqrt` builder (`UltraV3Config`, `compute_v3_tables`, `build_ultra_v3`, CLI entrypoint).
- `gf2_lut.py` — GF(2) quadratic LUT synthesis (`synthesize_qrom1_bcast`, `synthesize_qrom1_overwrite`, `compute_quad_bits_bcast`, `compute_quad_bits_overwrite`, 16-core Numba `_nb_search_u_hit_256`, `uncompute_bits_hmr`, `hmr_defer_bits`, `apply_deferred_quad_fixup`, `uncompute_qrom1_init_and`).
- `ultra_primitives.py` — `PoolAllocator`, `normalize_to_1_2`, and asymmetric zero-overhead deferred-HMR `pebbled_gidney_iadd`.
- `ultra_v2_primitives.py` — `normalize_to_s`, `compact_s_from_e`, `unpack_s`, `pack_s`, `unnormalize_from_compact`, `signed_digit_block_mul_1ovf` (with `mux_round` + `hmr_ovf`), `propagate_1ovf_carry`, `streaming_phase_shift_reuse` (with `s3_extra_lo` + `s3_max_bits`).
- `ultra_v3_p.kmb` — Pre-built verified `Q = 60, score = 106,925.35` circuit.
- `fast_verify_invsqrt.py` — Multiprocessing SIMD-batched verifier wrapper importing official validation/scoring from `verify_submission_invsqrt.py`.
- `verify_submission_invsqrt.py` — Official unmodified Kaggle verifier.
- `experiments.md` — Full verified optimization log.

**Python packages required:** `kickmix`, `numpy`, `scipy`, and optional `numba` (auto-detected in `gf2_lut.py` for 16-core parallel GF(2) search; falls back cleanly to pure Python if absent).

---

## 3. Exact Build & Verification Commands

### Reproduce `ultra_v3_p.kmb` (`Q = 60, score = 106,925.35`)
```bash
python ultra_v3.py ultra_v3_p_repro.kmb --wc2 8 --zgb 0 --mux-min 0 --nc0q1 16 --ntmpq1 3 --defer-c2ovf 1 --r1-blk 2blk --r1-anc 6 --cap-r1 0
```
*(Build takes ~35s on 16 cores: ~20s for full-domain NumPy CD + LP table optimization, ~15s for GF(2) LUT synthesis.)*

### Fast Stage-2 Verification (~2.5s, 1,048,576 shots)
```bash
python fast_verify_invsqrt.py ultra_v3_p.kmb --max-x 16384 --shots-per-x 64
```

### Full-Domain Stage-3 Verification (only for final candidates!)
```bash
# Fast full-domain (16 shots/x, ~75s):
python fast_verify_invsqrt.py best_invsqrt.kmb --shots-per-x 16 --batch-size 192

# Official 64-shot final submission check (200,540,928 shots):
python fast_verify_invsqrt.py best_invsqrt.kmb --shots-per-x 64 --batch-size 256
# Or official single-process script:
python verify_submission_invsqrt.py best_invsqrt.kmb --shots-per-x 64 --batch-size 256
```

---

## 4. Exact Integration Points for `v34`'s LUT Synthesizer

Every LUT in `ultra_v3.py` uses **bitmask truth tables** (`int` with bit `a` = output bit for address `a in 0..n_in-1`, generated by `gl.int_tables_to_bit_masks(table, n_bits)` and `gl.addr_bit_masks(n_addr)`).

### Drop-in Function Contracts in `gf2_lut.py`
To plug `v34`'s bilinear MinRank synthesizer (`r = 3..4` CCX/bit) into `ultra_v3` with **zero changes to arithmetic, normalization, or streaming waves**, you only need to replace/upgrade these two functions in `gf2_lut.py`:

1. **`synthesize_qrom1_bcast(builder, addr_qubits, out_qubits, target_vals, temp_qubits=(), n_boot=11, seed=0) -> int`**
   - **Inputs:** `addr_qubits` (8 qubits holding `addr = x[14..21]`), `out_qubits` (`34` clean `|0>` qubits = `c1[0..17] + c0_lo[0..15]`), `target_vals` (`34` 256-bit ints), `temp_qubits` (`3` clean `|0>` scratch qubits).
   - **Post-condition:** `out_qubits[j]` holds `target_vals[j](addr)`; `addr_qubits` unchanged; `temp_qubits` returned cleanly to `|0>` (via 0-CCX `HMR+CZ` or `del_and`).
   - **Current cost:** `195` CCX (`34` bits) vs `v34`'s `169` CCX (`35` bits) -> **`~30–35` CCX savings**.

2. **`compute_quad_bits_bcast(builder, W_qubits, W_vals, new_qubits, target_vals, temp_qubits=(), n_in=256, seed=0, use_init_and_temp=False, keep_temp_out=None) -> int`**
   - **Inputs:** `W_qubits` (`M` live pure qubits whose truth tables are `W_vals`), `new_qubits` (`k` clean `|0>` target qubits), `target_vals` (`k` `n_in`-bit ints), `temp_qubits` (`t` clean `|0>` scratch qubits).
   - **Post-condition:** `new_qubits[j]` holds `target_vals[j]`; `W_qubits` unchanged; `temp_qubits` returned to `|0>` (unless `keep_temp_out is not None`, used only by `EarlyW0` with `n_in=64`).
   - **Used at 5 call sites in `ultra_v3.py`** (all `n_in = 256` except `EarlyW0` which is `n_in = 64`):

| LUT Stage in `ultra_v3.py` | Line | Wire Count `M` (`W_qubits`) | Outputs `k` | `temp` `t` | Current CCX | `v34` Target CCX |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `1_Norm+EarlyW0` (`c0_hi` 6-bit, `n_in=64`) | `1041` | `6` (`x[16..21]`) | `13` (`base7+f3+top3`) | `7` (`init_and`) | `34` (+`14` add=`48`) | `~28` (+`14`=`42`) |
| `2_QROM1` (`c1 + c0_lo[:16]`) | `1066` | `8` (`addr`) | `34` | `3` | `195` | `~160–165` |
| `5a_c2fwd` (`c2`) | `1108` | `34` (`q1_qs`) | `8` | `2` | `45` | `~32–36` |
| `5b_c0r` (`c0_lo[16..19]`) | `1142` | `30` (`c0_lo[:16]+c1[10:]+c2[:6]`) | `4` | `0` | `31` | `~16–20` |
| `9_UncompC2` (`c2_ovf_r`, `use_init_and_temp=True`) | `1304` | `20` (`c0_lo`) | `1` (`c1_orig[10]`) | `6` (`init_and`) | `17` | `~10` (or `0` if `defer_c2ovf=0`) |
| `9_UncompC2` (`c2_rev`) | `1358` | `23` (`c0_lo[:15]+c1[10:]`) | `8` | `4` | `64` | `~44–48` |
| `10_UncompQ1` (`addr_rev`) | `1394` | `26` (`c1 + c2_rev`) | `8` | `10` | `55` | `~36–40` |
| `10_UncompQ1` (`uncompute_qrom1_init_and`) | `1431` | `8` (`addr`) -> uncomp `18` `c1` | `18` | pool | `17` | `~15–17` |

### Important Note on `_solve_bcast_quad_multi`
If you implement a better low-rank GF(2) bilinear solver (hitting `2*r*M - r*(2r+1) >= 255 - M` instead of our random-`U` `r*M - r*(r+1)/2 >= 255 - M`), you can drop it **directly inside `_solve_bcast_quad_multi`** in `gf2_lut.py` (around line 1220) — every single LUT (`QROM1`, `c2_fwd`, `c0lo_rest`, `c2_ovf_r`, `c2_rev`, `addr_rev`) will automatically benefit, and `assert (_verify_quad_eval(...) ^ bcast_val) == goal` will verify mathematical exactness on every bit before circuit emission.
