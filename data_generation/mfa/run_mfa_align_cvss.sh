#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J mfa_align
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/results/mfa/logs/mfa_align_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/results/mfa/logs/mfa_align_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=10:00:00

set -e

# =========================
# Arguments
# =========================
LANG=$1        # e.g., es, fr, de
TGT_LANG=$2    # e.g., en
TTS=$3         # true/false: whether input audio is generated TTS audio
subset=$4      # e.g., train/dev/test if needed

DELAY=$(( SLURM_ARRAY_TASK_ID * 20 ))
echo "Delaying start for $DELAY seconds..."
sleep "$DELAY"

# =========================
# Dataset setup
# =========================
DATASET=<DATASET>     # e.g., covost_v2 or cvss

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

RESULTS_ROOT="${BASE_DIR}/${USERID}/results"
UNTAR_ROOT="${BASE_DIR}/${USERID}/untarred_data"
TMP_ROOT="${BASE_DIR}/${USERID}/tmp"

MFA_RESOURCE_DIR="<MFA_RESOURCE_DIR>"   # directory containing pretrained_models, global_config.yaml, joblib_cache

mkdir -p "${RESULTS_ROOT}/mfa/logs"

# =========================
# Language mapping
# =========================
map_lang() {
  case "$1" in
    fr) echo "fr-FR" ;;
    en) echo "en-US" ;;
    de) echo "de-DE" ;;
    es) echo "es-US" ;;
    *)  echo "$1" ;;
  esac
}

LANG_CODE=$(map_lang "$LANG")
TGT_LANG_CODE=$(map_lang "$TGT_LANG")

echo "Language code: $LANG_CODE"
echo "Target language code: $TGT_LANG_CODE"

sharded_manifests=sharded_manifests

if [[ -z "$LANG" || -z "$TGT_LANG" ]]; then
    echo "❌ Usage: sbatch run_mfa_align.sh <LANG> <TGT_LANG> <TTS=true|false> <SUBSET>"
    exit 1
fi

# =========================
# Select MFA acoustic/dictionary model
# =========================
if [[ "$LANG" == "es" ]]; then
    echo "This is a Spanish $LANG job."

    if [[ "$TTS" == true ]]; then
        MFA="english_mfa"
    else
        MFA="spanish_mfa"
    fi

elif [[ "$LANG" == "fr" ]]; then
    echo "This is a French $LANG job."

    if [[ "$TTS" == true ]]; then
        MFA="english_mfa"
    else
        MFA="french_mfa"
    fi

elif [[ "$LANG" == "de" ]]; then
    echo "This is a German $LANG job."

    if [[ "$TTS" == true ]]; then
        MFA="english_mfa"
    else
        MFA="german_mfa"
    fi

else
    echo "Unknown language job: $LANG"
    exit 1
fi

# =========================
# Input/output setup
# =========================
if [[ "$DATASET" == "covost_v2" ]]; then

    if [[ "$TTS" == true ]]; then
        RESULTS_DIR="${RESULTS_ROOT}/mfa/tts_outputs/${DATASET}/${LANG}_${TGT_LANG}"
        data_path="${UNTAR_ROOT}/tts_outputs/${DATASET}/${LANG}_${TGT_LANG}"
    else
        RESULTS_DIR="${RESULTS_ROOT}/mfa/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
        data_path="${UNTAR_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
    fi

elif [[ "$DATASET" == "cvss" ]]; then

    if [[ "$TTS" == true ]]; then
        RESULTS_DIR="${RESULTS_ROOT}/mfa/tts_outputs/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${subset}"
        data_path="${UNTAR_ROOT}/tts_outputs/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${subset}"
    else
        RESULTS_DIR="${RESULTS_ROOT}/mfa/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${subset}"
        data_path="${UNTAR_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${subset}"
    fi

else
    echo "❌ Unsupported DATASET: $DATASET"
    exit 1
fi

txt_field="text"

mkdir -p "$RESULTS_DIR"

# =========================
# Environment setup
# =========================
export HYDRA_FULL_ERROR=1
unset LOCAL_RANK
unset SLURM_NTASKS
unset CUDA_VISIBLE_DEVICES

echo "******* STARTING ********"

source ~/.bashrc
conda activate <MFA_CONDA_ENV>

echo "Using Python at: $(which python)"
echo "Dataset: $DATASET"
echo "Data path: $data_path"
echo "Results dir: $RESULTS_DIR"
echo "MFA model: $MFA"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "$data_path/$sharded_manifests"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found $num_manifests manifests."

# =========================
# Run MFA alignment
# =========================
for i in $(seq 0 $(( num_manifests - 1 ))); do
    manifest_file="${manifest_array[$i]}"
    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    export MFA_ROOT_DIR="${TMP_ROOT}/${DATASET}/${LANG}_${TGT_LANG}/${subset}/${manifest_number}"
    MFA_ROOT_DIR="${TMP_ROOT}/${DATASET}/${LANG}_${TGT_LANG}/${subset}/${manifest_number}"

    mkdir -p "$MFA_ROOT_DIR"

    if [ -f "$RESULTS_DIR/${manifest_number}/ctm/words.combined.ctm" ] || [ "$(ls -A "$RESULTS_DIR/${manifest_number}/mfa" 2> /dev/null)" ]; then
        echo "✅ Skipping manifest $manifest_number – already processed or in progress of being processed"
        continue
    else
        # To avoid errors with MFA when running jobs in parallel, each job should have its own temporary directory.
        if [ -d "$MFA_ROOT_DIR" ]; then
            rm -rf "$MFA_ROOT_DIR"
        fi

        mkdir -p "$MFA_ROOT_DIR"

        ln -sf "$MFA_RESOURCE_DIR/pretrained_models" "$MFA_ROOT_DIR/pretrained_models"
        cp "$MFA_RESOURCE_DIR/global_config.yaml" "$MFA_ROOT_DIR/global_config.yaml"
        cp -r "$MFA_RESOURCE_DIR/joblib_cache" "$MFA_ROOT_DIR"
        ln -sf /dev/null "$MFA_ROOT_DIR/command_history.yaml"

        echo "MFA root: $MFA_ROOT_DIR"
        echo "Stage 1: generate MFA files from manifest $manifest_number"

        python manifest2mfa.py \
            --manifest "$manifest_file" \
            --txt-field "$txt_field" \
            --output-dir "$data_path/$manifest_number" \
            --lang "$LANG"

        echo "Generated text in: $data_path/$manifest_number"

        mkdir -p "$RESULTS_DIR/${manifest_number}/ctm"
        mkdir -p "$RESULTS_DIR/${manifest_number}/mfa"

        echo "Stage 2: 🚀 generate MFA alignments for manifest: $manifest_number"

        mfa align \
            --clean \
            --final_clean \
            --single_speaker \
            --num_jobs 4 \
            --overwrite \
            --profile "job_$(date +%s)" \
            --temporary_directory "$MFA_ROOT_DIR" \
            --output_format long_textgrid \
            --no_cleanup_textgrids \
            "$data_path/$manifest_number" \
            "$MFA" \
            "$MFA" \
            "$RESULTS_DIR/${manifest_number}/mfa" \
            --beam 100 \
            --retry_beam 400

        echo "Generate CTM from TextGrid"

        mv "$data_path/$manifest_number"/*.txt "$RESULTS_DIR/${manifest_number}/mfa/"

        for f in "$RESULTS_DIR/${manifest_number}/mfa"/*.TextGrid; do
            echo "Generate CTM from TextGrid: $f"
            python txtgrid2ctm.py --txt-grid "$f"
        done

        echo "📂 Combine CTM files into single CTM: $RESULTS_DIR/${manifest_number}/mfa"

        CTM_DIR="$RESULTS_DIR/${manifest_number}/ctm"
        COMBINED_FILE="$CTM_DIR/words.combined.ctm"

        cat "$RESULTS_DIR/${manifest_number}/mfa"/*.ctm > "$COMBINED_FILE"

        if [ -s "$COMBINED_FILE" ]; then
            echo "✅ Combined CTM written to: $COMBINED_FILE"

            rm -rf "$RESULTS_DIR/${manifest_number}/mfa"
            rm -rf "$MFA_ROOT_DIR"

            echo "🗑️ Deleted MFA temporary directories"
        else
            echo "❌ Failed to create combined CTM in $CTM_DIR. Skipping deletion."
        fi
    fi
done

# =========================
# Post-processing check
# =========================
total_manifests=$(find -L "$data_path/$sharded_manifests" -type f -name "manifest_*.jsonl" | wc -l)
total_done=$(find "$RESULTS_DIR" -name "words.combined.ctm" | wc -l)

if [ "$total_manifests" -eq "$total_done" ]; then
    echo "✅ All manifests have been processed!"
else
    echo "⚠️ Not all manifests have been processed: $total_done of $total_manifests"
fi