#!/bin/bash

root_path=$1
dataset=$2
bucket=$3
skipme=$4

# bash compute_duration.sh /lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts riva_asr_es-US_v1.0 1 1
# Usage check
if [ $# -lt 4 ]; then
    echo "Usage: $0 <root_path> <bucket> <dataset> <skipme (0/1)>"
    exit 1
fi

if [ $skipme -eq 0 ]; then
    echo "▶ Counting all segments (including _skipme=0)..."
    echo "Processing path: $root_path/$dataset/bucket${bucket}/sharded_manifests_es-US_en-US"
    
    total=$(cat $root_path/$dataset/bucket${bucket}/sharded_manifests_es-US_en-US/manifest_*.jsonl \
        | jq -r '.duration' \
        | awk '{s+=$1} END{print s}')


else
    echo "▶ Counting only valid segments (excluding _skipme=1)..."
    echo "Processing path: $root_path/$dataset/bucket${bucket}/sharded_manifests_es-US_en-US"
    total=$(cat $root_path/$dataset/bucket${bucket}/sharded_manifests_es-US_en-US/manifest_*.jsonl \
        | grep -v '"_skipme": 1' \
        | jq -r '.duration' \
        | awk '{s+=$1} END{print s}')
fi

hours=$(awk -v t="$total" 'BEGIN{printf "%.2f", t/3600}')

echo ""
echo "---------------------------------------------"
if [ "$skipme" -eq 0 ]; then
    echo "TOTAL CUT DURATION: $total seconds (${hours} hours)"
else
    echo "TOTAL CUT DURATION (skipme removed): $total seconds (${hours} hours)"
fi
echo "---------------------------------------------"