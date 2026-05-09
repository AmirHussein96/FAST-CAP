# Converting CVSS to NeMo-Compatible Data

## Step 1: Convert .mp3 to .wav 16 KHz

```
bash convert2wav.sh
```

## Step 2: Convert TSV metadata to NeMo manifests

```
bash tsv2manifest.sh 
```

## Step 3: Shard audio and manifests

```
nemo_root=<path_to_Nemo>
python $nemo_root/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
  --manifest_path <manifest_path>/manifest.jsonl \
  --target_dir <target_path> \
  --num_shards 40 \
  --max_duration 30
```