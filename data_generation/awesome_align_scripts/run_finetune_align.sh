#!/bin/bash
#SBATCH -A edgeai_riva_rivamlops
#SBATCH -J finetune_mbert
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/fsw/portfolios/edgeai/users/amhussein/results/awesome_align_clean/logs/alignments_%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/edgeai/users/amhussein/results/awesome_align_clean/logs/alignments_%A_%a.err
#SBATCH --partition=polar,polar3,polar4,interactive
#SBATCH --time=4:00:00


set -e

LANG=$1
TGT_LANG=$2
resume_train=$3
if [[ "$LANG" == "es" ]]; then
    echo "This is a Spanish $LANG job."
    DATASET=riva_asr_es-US_v1.0
    sharded_manifests=sharded_manifests_es-US_en-US 
    
elif [[ "$LANG" == "en" ]]; then
    echo "This is an English $LANG job."
    DATASET=riva_asr_en-US_v6.0
    if [[ "$TGT_LANG" == "de" ]]; then
        # MFA model to be used 
        sharded_manifests=sharded_manifests_en-US_de-DE
    fi
elif [[ "$LANG" == "fr" ]]; then
    echo "This is an French $LANG job."
    DATASET=riva_asr_fr-FR_v2.1
    sharded_manifests=sharded_manifests_fr-FR_en-US

elif [[ "$LANG" == "de" ]]; then
    echo "This is an German $LANG job."
    DATASET="riva_asr_de-DE_v1.0"
    sharded_manifests=sharded_manifests_de-DE_en-US
else
    echo "$LANG Unknown language job."
    exit 1
fi



export HF_HOME="/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
INPUT_BASE="/lustre/fsw/portfolios/edgeai/users/amhussein/results/awesome_align_clean/${DATASET}${TGT_LANG}"
OUTPUT_DIR=$INPUT_BASE
start_stage=1
stop_stage=1

. ~/.bashrc
conda activate awesome_align

# before running the finetuning combine all the files to train.parallel
# cat $INPUT_BASE/*/*.parallel > $INPUT_BASE/train.parallel
TRAIN_PAR="$OUTPUT_DIR/train.parallel"
DEV_PAR="$OUTPUT_DIR/dev.parallel"

if [[ ! -f "$TRAIN_PAR" ]]; then
    echo "❌ ERROR: Missing file $TRAIN_PAR"
    echo "Hint: Create it with:"
    echo "     cat $INPUT_BASE/*/*.parallel > $TRAIN_PAR"
    exit 1
fi

bash process_all_manifests.sh $INPUT_BASE "$OUTPUT_DIR" $start_stage $stop_stage $resume_train
