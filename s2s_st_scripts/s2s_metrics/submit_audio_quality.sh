#!/bin/bash
#SBATCH -A edgeai_riva_rivamlops
#SBATCH -J audio_quality
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --array=0-0
#SBATCH --partition=cpu,cpu_long,cpu_short
#SBATCH --time=10:00:00

set -euo pipefail
set -x

########################################
# User-configurable settings
########################################

# Optional positional arguments
LANG="${1:-}"
TGT_LANG="${2:-}"

# Conda
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-s2s-metrics}"

# Code
CODE_DIR="${CODE_DIR:-$PWD}"

# Logs
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "${LOG_DIR}"

# Audio input
# You can either pass AUDIO_DIR directly, or construct it from LANG/TGT_LANG below.
AUDIO_DIR="${AUDIO_DIR:-}"

# Max number of samples
MAX_SAMPLES="${MAX_SAMPLES:-2000}"

########################################
# Resolve audio directory
########################################

if [ -z "${AUDIO_DIR}" ]; then
    if [ -z "${LANG}" ] || [ -z "${TGT_LANG}" ]; then
        echo "ERROR: AUDIO_DIR is not set and LANG/TGT_LANG were not provided."
        echo ""
        echo "Usage option 1:"
        echo "  AUDIO_DIR=/path/to/wavs sbatch audio_quality_generic.sh"
        echo ""
        echo "Usage option 2:"
        echo "  DATA_ROOT=/path/to/data sbatch audio_quality_generic.sh fr-FR en-US"
        exit 1
    fi

    DATA_ROOT="${DATA_ROOT:-/path/to/data}"
    SPLIT="${SPLIT:-dev}"

    # Example constructed path:
    # /path/to/data/fr-FR_en-US/dev
    AUDIO_DIR="${DATA_ROOT}/${LANG}_${TGT_LANG}/${SPLIT}"
fi

########################################
# Slurm log files
########################################

OUTFILE="${LOG_DIR}/audio_quality_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.out"
ERRFILE="${LOG_DIR}/audio_quality_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.err"

exec > >(tee -a "${OUTFILE}") 2> >(tee -a "${ERRFILE}" >&2)

########################################
# Environment
########################################

if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: Could not find conda.sh at: ${CONDA_SH}"
    echo "Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "Using Python at: $(which python)"
echo "Python version: $(python --version)"
echo "Code dir: ${CODE_DIR}"
echo "Audio dir: ${AUDIO_DIR}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Language: ${LANG:-N/A}"
echo "Target language: ${TGT_LANG:-N/A}"

########################################
# Checks
########################################

cd "${CODE_DIR}"

if [ ! -d "${AUDIO_DIR}" ]; then
    echo "ERROR: Audio directory does not exist: ${AUDIO_DIR}"
    exit 1
fi

########################################
# Run metric
########################################

python audio_quality.py \
    --audio-dir "${AUDIO_DIR}" \
    --max-samples "${MAX_SAMPLES}"