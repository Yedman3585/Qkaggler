# dfthc_refit — Factorization Optimizer for dfthcsos track

Пайплайн оптимизации факторизации DF-THC для Benchmark III (Quantum Chemistry).
Цель: снизить lambda_eff с текущих 47.053 (ahoirg) до ≤44 для достижения 32× overall.

## Структура каталогов (на машине)

```
quantum_kaggle/
├── Qkaggler/                      # Git repo
│   ├── submission/                 # Текущий лучший сабмишн
│   ├── tools/                      # Утилиты (fastscore, angle_scan и т.д.)
│   └── optimizer/                  # ← ЭТОТ ПАКЕТ (dfthc_refit)
├── starter-kit/                   # Kaggle starter kit (содержит dfthc/)
│   └── starter_kit/
│       └── dfthc/
│           ├── df_thc_sga.py
│           └── train_df_thc_sga.py
└── competition_data/              # Данные конкурса
    └── verifiers/
        └── dfthcsos/
            ├── FeMoCo76o113e.fcidump     (~128 MB)
            └── FeMoCo76o113e.fixed_shifts.npz
```

## Настройка среды

```bash
source ~/venv-jax/bin/activate

# Установить переменные окружения (добавить в ~/.bashrc для постоянства):
export DFTHC_DATA="/mnt/c/Users/user/Desktop/quantum_kaggle/competition_data"
export DFTHC_STARTER="/mnt/c/Users/user/Desktop/quantum_kaggle/starter-kit/starter_kit"

# Проверка:
python -c "
import sys; sys.path.insert(0, '.')
from paths import DATA, STARTER, FCIDUMP
print('DATA:', DATA)
print('STARTER:', STARTER)
print('FCIDUMP exists:', FCIDUMP.exists())
"
```

## Пайплайн

### 1. Continuous fit (L-BFGS, ~часы на CPU)
```bash
cd /mnt/c/Users/user/Desktop/quantum_kaggle/Qkaggler/optimizer
python lbfgs_fit.py SEED.npz output.npz --iters 3000 --kappa 2e-5
```
SEED.npz — стартовая факторизация (можно взять public seed из starter kit).

### 2. Quantize (округление до brot бит)
```bash
python quantize.py output.npz rounded.npz --brot 8
```

### 3. Refine (оптимизация весов после округления)
```bash
python refine.py rounded.npz refined.npz --brot 8
```

### 4. Refit weights (финальная подгонка W при фиксированных словах)
```bash
python refit_weights.py refined.npz final.npz
```

### 5. Build & verify (собрать angle circuits + проверить)
```bash
python build.py final.npz submission_dir/ --brot 8
```

### Или всё сразу через recipe:
```bash
python run.py recipes/example.toml --out runs/experiment1
```

## Ключевые параметры для 32×

Текущее (ahoirg): R=9, B=143, C=42 → lambda_eff = 47.053
Нужно: lambda_eff ≤ 44

Варианты:
- Уменьшить R (rank): меньше lambda, но хуже fit → выше error
- Уменьшить B (bases): меньше NA → дешевле angle circuits
- Уменьшить C (copies): меньше lambda напрямую
- Улучшить fit (меньше error_1, error_2) при тех же R,B,C

## Зависимости

- JAX + jaxlib (CPU ok, GPU лучше)
- numpy, scipy
- optax (для L-BFGS и Adam)
- openfermion (для загрузки FCIDUMP)
- dfthc (из starter kit — через DFTHC_STARTER env var)
