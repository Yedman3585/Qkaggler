# `invsqrt` Optimization Experiments

| version | arch | Q | Toffoli | T | n_rot | C | score | max_err_bits |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | NR + Horner (4-case cubic) | 260 | 7735 | 1 | 37 | 7966.45 | 2,071,276 | -25.07 |
| quad_v1 | Quad QROM (k=8, wp=26, gb=3) | 164 | 4174 | 1 | 33 | 4380.05 | 718,328 | -24.87 |
| quad_v2 | Quad QROM + t_delta + padded iadd | 176 | 2796 | 1 | 33 | 3002.05 | 528,361 | -24.87 |
| quad_v3 | Quad QROM (3-split + t2 inv, gb=2) | 112 | 3832 | 1 | 33 | 4040.11 | 452,493 | -24.27 |
| quad_v4 | Horner Quad QROM (single, gb=3) | 150 | 2576 | 1 | 33 | 2783.19 | 417,478 | -24.85 |
| quad_v5 | Horner Quad QROM (3-split, wp=26, gb=2, d_c2=9) | 124 | 3284 | 1 | 33 | 3494.75 | 433,349 | -24.14 |
| quad_v5_single | Horner Quad QROM (single, wp=26, gb=2, d_c2=9) | 142 | 2352 | 1 | 33 | 2562.75 | 363,911 | -24.14 |
| best_invsqrt | Horner Quad QROM (single, wp=26, gb=2, wc2=12, d_c2=9) | 141 | 2352 | 1 | 33 | 2586.77 | 364,735 | -24.14 |
| ultra_v2_c | Compact s + GF(2) QROM + 2-wave streaming | 68 | 1736.0 | 1 | 44 | 2053.30 | 139,624 | -24.00 |
| ultra_v3_a | Unsplit c2 + PoolAllocator + defer_hmr | 63 | 1608.5 | 1 | 44 | 1919.99 | 120,959 | -24.00 |
| ultra_v3_c | BCAST LUTs + tuned r1_anc | 62 | 1564.5 | 1 | 44 | 1876.01 | 116,312 | -24.00 |
| ultra_v3_d | Asym defer-HMR c2*u + hmr_ovf + mux_min=0 | 61 | 1513.0 | 1 | 44 | 1824.50 | 111,295 | -24.00 |
| ultra_v3_f2 | Early 6-bit W0 + s3==1 fusion + wc2=9 zgb=1 | 61 | 1509.0 | 1 | 41 | 1798.73 | 109,723 | -24.00 |
| ultra_v3_i | Early 6-bit W0 + s3==1 + A_min staircase fix | 61 | 1501.5 | 1 | 41 | 1791.20 | 109,263 | -24.00 |
| ultra_v3_j | Early 6-bit W0 + s3==1 + wc2=8 zgb=1 | 61 | 1497.5 | 1 | 41 | 1791.02 | 109,252 | -24.00 |
| ultra_v3_k | Q=60 lock (mux_min=4) | 60 | 1493.5 | 1 | 41 | 1799.36 | 107,961 | -24.00 |
| ultra_v3_k2 | Q=60 + 16-core Numba max_def=13 LUTs | 60 | 1481.5 | 1 | 41 | 1787.33 | 107,240 | -24.00 |
| ultra_v3_p | Q=60 + skip_mux_23 + nc0q1=16 + Numba max_def=13 | 60 | 1481.5 | 1 | 41 | 1782.09 | 106,925 | -24.00 |
