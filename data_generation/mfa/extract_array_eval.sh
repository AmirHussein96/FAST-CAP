#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J es_extract
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/untarred_data/logs/extract_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/untarred_data/logs/extract_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=10:00:00

set -e

# =========================
# Arguments
# =========================
LANG=$1
TGT_LANG=$2
TTS=$3
DATASET=<DATASET>   # e.g., covost_v2 or cvss
subset=$4

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

DATA_ROOT="${BASE_DIR}/${USERID}/data"
RESULTS_ROOT="${BASE_DIR}/${USERID}/results"
UNTAR_ROOT="${BASE_DIR}/${USERID}/untarred_data"

mkdir -p "${UNTAR_ROOT}/logs"

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
echo "This is a $LANG job."

sharded_manifest=sharded_manifests

# =========================
# Input/output setup
# =========================
if [[ "$DATASET" == "covost_v2" ]]; then

    if [[ "$TTS" == true ]]; then
        INPUT_BASE="${RESULTS_ROOT}/a2flow_tts/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
        TARGET_BASE="${UNTAR_ROOT}/a2flow_tts/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
    else
        INPUT_BASE="${DATA_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
        TARGET_BASE="${UNTAR_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"
    fi

elif [[ "$DATASET" == "cvss" ]]; then

    if [[ "$TTS" == true ]]; then
        INPUT_BASE="${RESULTS_ROOT}/a2flow_tts/${DATASET}/${LANG_CODE}/${subset}"
        TARGET_BASE="${UNTAR_ROOT}/a2flow_tts/${DATASET}/${LANG_CODE}/${subset}"
    else
        INPUT_BASE="${DATA_ROOT}/${DATASET}/${LANG_CODE}/nemo_prepared_${subset}"
        TARGET_BASE="${UNTAR_ROOT}/${DATASET}/${LANG_CODE}/${subset}"
    fi

else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# =========================
# Create target directories
# =========================
echo "target base: $TARGET_BASE"
mkdir -p "$TARGET_BASE"

echo "Processing Dataset: $DATASET"

bucket_name=""
BUCKET_DIR="$INPUT_BASE"

if [[ -z "$bucket_name" ]]; then
    echo "Bucket name is empty"
else
    echo "Bucket name is: $bucket_name"
fi

# =========================
# Collect tar files
# =========================
shuffled_audios=$(ls "$BUCKET_DIR"/audio_*.tar | shuf)
audio_array=($shuffled_audios)
num_audios=${#audio_array[@]}

mkdir -p "$TARGET_BASE/${bucket_name}"

# =========================
# Extract files
# =========================
for i in $(seq 0 $(( num_audios - 1 ))); do
    input_file="${audio_array[$i]}"

    echo "file: $input_file"

    filename=$(basename "$input_file")
    audio_num="${filename%.*}"       # removes .tar
    audio_num="${audio_num#audio_}"  # removes audio_

    if [[ -z "$bucket_name" ]]; then
        target_folder="$TARGET_BASE/${audio_num}"
    else
        target_folder="$TARGET_BASE/${bucket_name}/${audio_num}"
    fi

    # Skip extraction if the target folder exists and is not empty
    if [ -d "$target_folder" ] && [ "$(ls -A "$target_folder")" ]; then
        echo "📌 Skipping $input_file: $target_folder already contains files."
        continue
    fi

    mkdir -p "$target_folder"

    tar -xf "$input_file" -C "$target_folder"

    echo "Processed $input_file in $target_folder"

    if [ -L "$TARGET_BASE/${bucket_name}/$sharded_manifest" ]; then
        echo "Symlink exists"
    else
        ln -sf "$BUCKET_DIR/$sharded_manifest" "$TARGET_BASE/${bucket_name}/$sharded_manifest"
    fi

    # Check if number of extracted files is equal to number of files in the tar file
    num_extracted_files=$(ls -1 "$target_folder"/ | wc -l)
    num_files_in_tar=$(tar -tf "$input_file" | wc -l)

    if [ "$num_extracted_files" -ne "$num_files_in_tar" ]; then
        echo "❌ Error with file extraction for $input_file"
        echo "❌ Error: Number of extracted files ($num_extracted_files) does not match number of files in tar file ($num_files_in_tar)"
        rm -rf "$target_folder"
    fi
done

echo "✅ Finished processing all files"
echo "✅ Script completed successfully"