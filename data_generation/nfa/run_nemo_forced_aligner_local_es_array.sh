#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J nemo_align_es
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-10   # Adjust based on number of buckets
#SBATCH --output=nemo_align_%A_%a.out
#SBATCH --error=nemo_align_%A_%a.err
#SBATCH --partition=<PARTITION>
#SBATCH --time=4:00:00

set -euo pipefail

# =========================
# Language/model setup
# =========================
if [[ "$SLURM_JOB_NAME" == *"es"* ]]; then
    echo "This is a Spanish (es) job."
    DATASET="riva_asr_es-US_v1.0"
    MANIFEST_SUBDIR="sharded_manifests_es-US_en-US"
    MODEL_PATH="<PATH_TO_SPANISH_NEMO_MODEL>"
elif [[ "$SLURM_JOB_NAME" == *"en"* ]]; then
    echo "This is an English (en) job."
    DATASET="riva_asr_en-US_v6.0"
    MANIFEST_SUBDIR="sharded_manifests_en-US"
    MODEL_PATH="<PATH_TO_ENGLISH_NEMO_MODEL>"
else
    echo "Unknown language job: ${SLURM_JOB_NAME}"
    exit 1
fi

# =========================
# Array bucket setup
# =========================
# If your buckets are bucket1, bucket2, ..., bucket11:
bucket=$((SLURM_ARRAY_TASK_ID + 1))

# If your buckets are bucket0, bucket1, ..., bucket10, use this instead:
# bucket=${SLURM_ARRAY_TASK_ID}

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing bucket: ${bucket}"

# =========================
# Generic paths
# =========================
USERID="<USERID>"
BASE_DIR="<BASE_DIR>"

CODE_DIR="${BASE_DIR}/${USERID}/toolkits/NeMo"
DATA_DIR="${BASE_DIR}/${USERID}/results/a2flow_tts/${DATASET}/bucket${bucket}"
RESULTS_DIR="${BASE_DIR}/${USERID}/results/nemo_forced_aligner_a2flow/${DATASET}/bucket${bucket}"

MANIFEST_DIR="${DATA_DIR}/${MANIFEST_SUBDIR}"

mkdir -p "${RESULTS_DIR}/logs"

# =========================
# Environment setup
# =========================
export HYDRA_FULL_ERROR=1

# Usually safer not to unset CUDA_VISIBLE_DEVICES.
# Slurm sets it to the GPU assigned to the job.
unset LOCAL_RANK
unset SLURM_NTASKS

echo "******* STARTING ********"

source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Using model path: ${MODEL_PATH}"
echo "Code dir: ${CODE_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Manifest dir: ${MANIFEST_DIR}"

# =========================
# Check manifest files
# =========================
if ! compgen -G "${MANIFEST_DIR}/manifest_*.jsonl" > /dev/null; then
    echo "❌ No manifest files found in ${MANIFEST_DIR}"
    exit 1
fi

mapfile -t manifest_array < <(find "${MANIFEST_DIR}" -name "manifest_*.jsonl" | sort | shuf)

num_manifests=${#manifest_array[@]}
echo "Found ${num_manifests} manifests."

# =========================
# Run alignment
# =========================
for i in "${!manifest_array[@]}"; do
    manifest_file="${manifest_array[$i]}"

    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')
    tar_path="${DATA_DIR}/audios_${manifest_number}.tar"
    output_dir="${RESULTS_DIR}/${manifest_number}"

    mkdir -p "${output_dir}"

    if [ -f "${output_dir}/ctm/words.combined.ctm" ]; then
        echo "✅ Skipping manifest ${manifest_number} — already processed."
        continue
    fi

    if [ ! -f "${tar_path}" ]; then
        echo "❌ Missing tar file: ${tar_path}"
        exit 1
    fi

    echo "🚀 Running alignment for manifest ${manifest_number}"
    echo "Manifest file: ${manifest_file}"
    echo "Tar path: ${tar_path}"
    echo "Output dir: ${output_dir}"

    python "${CODE_DIR}/tools/nemo_forced_aligner/align.py" \
        batch_size=64 \
        model_path="${MODEL_PATH}" \
        manifest_filepath="${manifest_file}" \
        save_output_file_formats="['ctm']" \
        output_dir="${output_dir}" \
        load_lhotse_tarred=True \
        combine_ctms=True \
        tar_path="${tar_path}"

    if [ -f "${output_dir}/ctm/words.combined.ctm" ]; then
        echo "✅ Done with manifest ${manifest_number} → ${output_dir}"
    else
        echo "❌ Error: No CTM file found for manifest ${manifest_number}"
        exit 1
    fi
done

# =========================
# Post-processing check
# =========================
total_manifests=$(find "${MANIFEST_DIR}" -name "manifest_*.jsonl" | wc -l)
total_done=$(find "${RESULTS_DIR}" -name "words.combined.ctm" | wc -l)

if [ "${total_manifests}" -eq "${total_done}" ]; then
    echo "✅ All manifests have been processed for bucket ${bucket}!"
else
    echo "⚠️ Not all manifests have been processed for bucket ${bucket}: ${total_done} of ${total_manifests}"
fi