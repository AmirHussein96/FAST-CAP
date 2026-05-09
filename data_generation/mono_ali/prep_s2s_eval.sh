# modify the audio path in the jsonl file

sed 's|/data/covost_v2/wav|/lustre/fsw/portfolios/llmservice/users/amhussein/data/validation/audio/covost_v2|g'     /lustre/fsw/portfolios/llmservice/users/amhussein/data/validation/v2/covost_v2/dev/covost_v2.fr-FR_en-US.jsonl > covost_v2.fr-FR_en-US.jsonl

# if the file is large split to shars:
python /lustre/fsw/portfolios/edgeai/users/amhussein/toolkits/NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py --manifest_path covost_v2.fr-FR.jsonl --target_dir /lustre/fsw/portfolios/edgeai/users/amhussein/data/covost_v2/fr-FR_en-US --num_shards 40 --max_duration 30

# change the extension from json files to jsonl
cd /lustre/fsw/portfolios/llmservice/users/amhussein/data/covost_v2/de-DE_en-US/sharded_manifests

for f in *.json; do
  mv "$f" "${f%.json}.jsonl"
done

sed 's|/data/covost_v2/wav|/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/AST/data/validation/audio/covost_v2/|g'     covost_v2.fr-FR_en-US.jsonl > covost_v2.fr-FR_en-US2.jsonl