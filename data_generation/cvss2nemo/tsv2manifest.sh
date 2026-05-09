#!/bin/bash

set -e

langs=(
    "es_en"
    "fr_en"
    "de_en"
)
subsets=(
    "dev"
    "test"
    "train"
)
cvss_root=data/cvss
echo "CVSS root: $cvss_root"
echo "Processing CVSS data"
for lang in "${langs[@]}"; do
    for subset in "${subsets[@]}"; do
        manifest_dir=$cvss_root/manifest_${lang}_${subset}.jsonl
        if [ -f "$manifest_dir" ]; then
            echo "Manifest already exists for $lang $subset"
            continue
        else
            echo "Processing $lang $subset"
            python tsv2manifest.py \
                --lang $lang \
                --origin_tsv $cvss_root/origin/$lang/$subset.tsv \
                --translated_tsv $cvss_root/$lang/$subset.tsv \
                --source-audio-root $cvss_root/origin \
                --output-dir $cvss_root \
                --subset $subset
            echo "Done $lang $subset"
        fi
    done
done