#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J extract_data
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

DATASET=<DATASET>      # e.g., asr_fr-FR_v1.0, asr_es-US_v1.0, speech_v1.0
bucket_name=bucket2    # change as needed

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

PROJECT_DATA_ROOT="${BASE_DIR}/projects/<PROJECT_NAME>/data/<DATA_SUBDIR>"
USER_DATA_ROOT="${BASE_DIR}/${USERID}/data"
RESULTS_ROOT="${BASE_DIR}/${USERID}/results"
UNTAR_ROOT="${BASE_DIR}/${USERID}/untarred_data"

mkdir -p "${UNTAR_ROOT}/logs"

# =========================
# Helper functions
# =========================
get_lang_code_from_dataset() {
    local dataset="$1"

    if [[ "$dataset" =~ asr_([a-z]{2})-[A-Z]{2}_v[0-9.]+ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$LANG"
    fi
}

capitalize_lang() {
    local lang="$1"
    echo "${lang^}"
}

map_lang_region() {
    case "$1" in
        fr) echo "fr-FR" ;;
        en) echo "en-US" ;;
        de) echo "de-DE" ;;
        es) echo "es-US" ;;
        *)  echo "$1" ;;
    esac
}

DATASET_LANG=$(get_lang_code_from_dataset "$DATASET")
DATASET_LANG_CAP=$(capitalize_lang "$DATASET_LANG")
DATASET_LANG_REGION=$(map_lang_region "$DATASET_LANG")

echo "Dataset: $DATASET"
echo "Language from dataset: $DATASET_LANG"
echo "Capitalized language: $DATASET_LANG_CAP"
echo "Language region: $DATASET_LANG_REGION"

# =========================
# Language/data setup
# =========================
if [[ "$DATASET_LANG" == "es" ]]; then
    echo "This is a ${DATASET_LANG_CAP} job."
    sharded_manifest=sharded_manifests_es-US_en-US

    if [[ "$DATASET" == "speech_v1.0" ]]; then

        if [[ "$TTS" == true ]]; then
            INPUT_BASE="${RESULTS_ROOT}/tts_outputs/${DATASET}es-US"
            TARGET_BASE="${UNTAR_ROOT}/tts_outputs/${DATASET}es-US"
        else
            INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}/es-US"
            TARGET_BASE="${UNTAR_ROOT}/${DATASET}/es-US"
        fi

    elif [[ "$DATASET" == "asr_es-US_v1.0" ]]; then

        if [[ "$TTS" == true ]]; then
            INPUT_BASE="${RESULTS_ROOT}/tts_outputs/${DATASET}${TGT_LANG}"
            TARGET_BASE="${UNTAR_ROOT}/tts_outputs/${DATASET}${TGT_LANG}"
        else
            INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}"
            TARGET_BASE="${UNTAR_ROOT}/${DATASET}${TGT_LANG}"

            # Alternative original-data style paths:
            # INPUT_BASE="${USER_DATA_ROOT}/train/audio/asr_es-US_v1.0"
            # TARGET_BASE="${UNTAR_ROOT}/original_data/asr_es-US_v1.0"
        fi
    fi

elif [[ "$DATASET_LANG" == "en" ]]; then
    echo "This is a ${DATASET_LANG_CAP} job with target language $TGT_LANG."

    INPUT_BASE="${PROJECT_DATA_ROOT}/asr_en-US_v6.0"

    if [[ "$TGT_LANG" == "de" ]]; then
        TARGET_BASE="${UNTAR_ROOT}/asr_en-US_de-DE_v6.0"
        sharded_manifest=sharded_manifests_en-US_de-DE
    fi

elif [[ "$DATASET_LANG" == "de" ]]; then
    echo "This is a ${DATASET_LANG_CAP} job."
    sharded_manifest=sharded_manifests_de-DE_en-US

    if [[ "$DATASET" == "speech_v1.0" ]]; then

        if [[ "$TTS" == true ]]; then
            INPUT_BASE="${RESULTS_ROOT}/tts_outputs/${DATASET}de-DE"
            TARGET_BASE="${UNTAR_ROOT}/tts_outputs/${DATASET}de-DE"
        else
            INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}/de-DE"
            TARGET_BASE="${UNTAR_ROOT}/${DATASET}/de-DE"
        fi

    elif [[ "$DATASET" == "asr_de-DE_v1.0" ]]; then
        INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}"
        TARGET_BASE="${UNTAR_ROOT}/${DATASET}"
    fi

elif [[ "$DATASET_LANG" == "fr" ]]; then
    echo "This is a ${DATASET_LANG_CAP} job."
    sharded_manifest=sharded_manifests_fr-FR_en-US

    if [[ "$DATASET" == "speech_v1.0" ]]; then

        if [[ "$TTS" == true ]]; then
            INPUT_BASE="${RESULTS_ROOT}/tts_outputs/${DATASET}fr-FR"
            TARGET_BASE="${UNTAR_ROOT}/tts_outputs/${DATASET}fr-FR"
        else
            INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}/fr-FR"
            TARGET_BASE="${UNTAR_ROOT}/${DATASET}/fr-FR"
        fi

    elif [[ "$DATASET" == "asr_fr-FR_v1.0" ]]; then

        if [[ "$TTS" == true ]]; then
            INPUT_BASE="${RESULTS_ROOT}/tts_outputs/${DATASET}${TGT_LANG}"
            TARGET_BASE="${UNTAR_ROOT}/tts_outputs/${DATASET}${TGT_LANG}"
        else
            INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}"
            TARGET_BASE="${UNTAR_ROOT}/${DATASET}"
        fi
    fi
fi

# =========================
# Safety check
# =========================
if [[ -z "${INPUT_BASE:-}" ]]; then
    echo "❌ INPUT_BASE was not set. Check DATASET=$DATASET and DATASET_LANG=$DATASET_LANG"
    exit 1
fi

if [[ -z "${TARGET_BASE:-}" ]]; then
    echo "❌ TARGET_BASE was not set. Check DATASET=$DATASET and DATASET_LANG=$DATASET_LANG"
    exit 1
fi

if [[ -z "${sharded_manifest:-}" ]]; then
    echo "❌ sharded_manifest was not set. Check DATASET=$DATASET and DATASET_LANG=$DATASET_LANG"
    exit 1
fi

# =========================
# Create target directories
# =========================
echo "target base: $TARGET_BASE"
mkdir -p "$TARGET_BASE"

echo "Processing Dataset: $DATASET"

if [[ "$DATASET" == asr* ]]; then
    echo "Processing bucket: $bucket_name"
    BUCKET_DIR="$INPUT_BASE/$bucket_name"
else
    bucket_name=""
    BUCKET_DIR="$INPUT_BASE"
fi

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
# Extract tar files
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

    num_extracted_files=$(ls -1 "$target_folder"/ | wc -l)
    num_files_in_tar=$(tar -tf "$input_file" | wc -l)

    if [ "$num_extracted_files" -ne "$num_files_in_tar" ]; then
        echo "❌ Error with file extraction for $input_file"
        echo "❌ Error: Number of extracted files ($num_extracted_files) does not match number of files in tar file ($num_files_in_tar)"
        rm -rf "$target_folder"
    fi
done

echo "✅ Finished processing all files in bucket $bucket_name"
echo "✅ Script completed successfully"