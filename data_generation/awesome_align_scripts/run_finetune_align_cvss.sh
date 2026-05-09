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
DATASET=cvss
subset=$3
sharded_manifests=sharded_manifests


echo "Processing ${DATASET} ${LANG} ${TGT_LANG} ${subset}"

export HF_HOME="/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
INPUT_BASE="/lustre/fsw/portfolios/edgeai/users/amhussein/results/awesome_align_clean/${DATASET}/${LANG}_${TGT_LANG}/$subset"
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
