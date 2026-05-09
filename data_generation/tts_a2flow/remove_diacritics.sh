#!/bin/bash
#SBATCH -A edgeai_riva_rivamlops
#SBATCH -J check_ascii
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=process_manifests_%A.out
#SBATCH --error=process_manifests_%A.err
#SBATCH --partition=cpu
#SBATCH --time=8:00:00


bucket=1

dir2=/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/riva_asr_es-US_v1.0/bucket${bucket}/sharded_manifests_es-US_en-US


. ~/.bashrc
conda activate nemo
echo "Using Python at: $(which python)"

shuffled_manifests=$(ls $dir2/manifest_*.jsonl | shuf)
#shuffled_manifests=$(ls $dir2/manifest_0.jsonl)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

for i in $(seq 0 $num_manifests); do
    
    manifest_file=${manifest_array[$i]}
    echo processing bucket $bucket, manifest $manifest_file 
    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')
    manifest="$dir2/manifest_${manifest_number}.jsonl"
     
    python remove_diacritics.py --manifest "$manifest"
done