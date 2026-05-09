#!/bin/bash
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J check_ascii
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
##SBATCH --array=0-0  # Set this to the number of buckets you have
#SBATCH --output=process_manifests_%A_%a.out
#SBATCH --error=process_manifests_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --time=8:00:00


bucket=1

dir1=/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/AST/data/train/v2/riva_asr_es-US_v1.0/bucket${bucket}/sharded_manifests_es-US_en-US
dir2=/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/riva_asr_es-US_v1.0/bucket${bucket}/sharded_manifests_es-US_en-US


. ~/.bashrc
conda activate nemo
echo "Using Python at: $(which python)"

shuffled_manifests=$(ls $dir2/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

processed_count=0
error_count=0

for i in $(seq 0 $num_manifests); do
    manifest_file=${manifest_array[$i]}
    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')
    ref_manifest="$dir1/manifest_${manifest_number}.jsonl"
     
    if python check_ascii.py --manifest1 "$ref_manifest" --manifest2 "$manifest_file"; then
        echo "  ✓ Successfully processed manifest_$manifest_number"
        processed_count=$((processed_count + 1))
    else
        echo "  ✗ Error processing manifest_$manifest_number"
        error_count=$((error_count + 1))
    fi
done

echo "================================================================"
echo "Summary:"
echo "  Successfully processed: $processed_count"
echo "  Errors: $error_count"