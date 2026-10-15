#!/bin/bash
# Preprocess all archived HDF5 shots into HISTORICAL_DATA_PATH (CPU only).
#PBS -N dldl_preprocess_historical
#PBS -l select=1:system=polaris,place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe

set -euo pipefail

cd "$PROJECT_ROOT"

# Load every setting from .env into the environment.
set -a
source .env
set +a

# Activate the conda environment that has PyTorch etc.
source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

python src/preprocess_historical.py
