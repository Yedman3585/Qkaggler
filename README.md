# Qkaggler — Google Quantum AI: Verifiable Quantum Circuit Compilation

Kaggle competition: [Verifiable Quantum Circuit Compilation](https://www.kaggle.com/competitions/google-quantum-ai-verifiable-circuit-compilation)

**Prize**: $150K | **Deadline**: Jan 20, 2027

## Current Best: 23.85×

| Track | Circuit | Score | Factor | Status |
|-------|---------|-------|--------|--------|
| invsqrt | v34 (inzainos) | 97,921.63 | **21.21×** | PASS ✓ |
| DQI | wahyu hybrid6-guard893 (Q=893) | 387,059,384,920 | **63.14×** | PASS ✓ |
| dfthcsos | ahoirg (J=5, K=15, brot=8) | 331,076,722 | **10.14×** | PASS ✓ |

**Overall** = (21.21 × 63.14 × 10.14)^(1/3) = **23.85×**

## Analysis Summary

### Track 1 — invsqrt (Fusion Energy)
- v34 beats ahoirg by 2.7% (21.21× vs 20.65×)
- Q=62, mean_toffoli=1281.67, rotations=41

### Track 2 — DQI (Combinatorial Optimization)
- **Ceiling reached at 63.14×**
- Q=888 attempt: mean_toffoli +7.7% → 59.27× (worse)
- Q=892 attempt: failed Kaggle draw (2 errors)
- No further DQI improvement possible

### Track 3 — dfthcsos (Quantum Chemistry)
- ahoirg factorization: R=9, B=143, C=42, NA=1363, lambda_eff=47.053
- **32× is mathematically impossible** with this factorization:
  - Even with zero angle Toffoli: max dfthcsos = 24.09×
  - Max overall = (21.21 × 63.14 × 24.09)^(1/3) = 31.83×
- J-optimization analysis: J>5 needs QROAM with limited ancilla (unresolved)
- **Path to 32×**: new factorization with lambda_eff ≤ 44 (requires JAX + dfthc)

## Repository Structure

```
submission/           # Current best submission
  submission_v34.zip  # Kaggle-ready zip (without dqi.kmb — too large for git)
  invsqrt.kmb         # v34 invsqrt circuit
  dfthcsos/           # ahoirg dfthcsos circuits + factorization
    angle_k_*.kmb     # 15 angle circuits
    submission.npz    # Factorization data
tools/                # Analysis & optimization tools
  dfthc_fastscore.py  # Fast dfthcsos score evaluator
  angle_scan.py       # Angle circuit redundancy scanner
  full_refit.py       # PyTorch joint angle+weight optimizer (abandoned — doesn't converge)
```

**Note**: `dqi.kmb` (550 MB) is not in the repo — download from wahyu's submission or rebuild from source.

## Key Formulas

- **Overall Score** = (REF_inv/cost_inv × REF_dqi/cost_dqi × REF_dfthc/cost_dfthc)^(1/3)
- **DQI Score** = R(eps) × mean_toffoli × Q²
- **dfthcsos Score** = lambda_eff × T_total × Q_total
  - T_total = 2 × K × tree_toffoli + heuristic_toffoli
  - Q_total = 388 + J × brot (for ahoirg params)
