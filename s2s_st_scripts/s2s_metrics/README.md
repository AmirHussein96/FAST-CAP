# S2S Metrics

This folder contains scripts and utilities for **evaluating Speech-to-Speech Speech Translation (S2S-ST)** systems.

The evaluation covers:

- **Translation quality** using BLEU, chrF, and COMET
- **Speaker similarity**
- **Audio quality**

---

## Environment Setup

The full evaluation environment is provided through:

- `Python 3.10.19`
- `requirements.txt`
- `environment.yaml`

The main required packages are:

- `speechbrain`: for speaker similarity evaluation
- `unbabel-comet`: for COMET scoring
- `sacrebleu`: for BLEU and chrF scoring

You can recreate the environment using Conda:

```bash
conda env create -f environment.yaml
conda activate <env-name>
```


## Translation Quality Scoring

The translation quality script evaluates the generated S2S-ST outputs using text-based metrics such as **BLEU**, **chrF**, and optionally **COMET**.

### 1. Download the COMET model

If you want to compute COMET scores, first download a COMET checkpoint.

```bash
comet-download -m Unbabel/wmt22-comet-da
```
After downloading the model, pass the checkpoint path to the scoring script using:

```bash
--comet-ckpt <path-to-comet-checkpoint>
```

For example:

```bash
python score_st.py \
  --json validation_logs/metadatas/cvss_test.json \
  --use-comet \
  --comet-ckpt ~/.cache/torch/unbabel_comet/Unbabel/wmt22-comet-da/checkpoints/model.ckpt
```

### 2. Run the scoring script

To include COMET scoring:

```bash
python score_st.py \
  --json <path-to-generated-results.json> \
  --use-comet \
  --lower \
  --rm-punctuation \
  --clean-english-text \
  --comet-ckpt <path-to-comet-checkpoint>
```

#### Expected JSON format:

Each entry in the JSON file should follow this format:

```bash
{
    "target_text": "Its ventral side is pale green, marbled with white.",
    "pred_text": "Its ventral face is pale green marbled with white.",
    "speech_pred_transcribed": "its ventral face is pale green marble with white",
    "audio_path": "pred_wavs/common_voice_fr_19841402.wav"
}
```

Where:
- `target_text`: reference translation
- `pred_text`: predicted text translation
- `speech_pred_transcribed`: ASR transcription of generated speech
- `audio_path`: path to the generated audio waveform


### Speaker Similarity Evaluation

To compute speaker similarity between source and generated speech:

```bash
python spk_sim.py \
  --source-audio-dir <path-to-source-audio> \
  --target-audio-dir <path-to-generated-audio>
```

This script compares speaker characteristics between the source and target audio directories.

### Audio Quality Evaluation

```bash
sbatch submit_audio_quality.sh <source-lang> <tgt-lang>
```

#### Notes

- `<source-lang>` and `<tgt-lang>` are only used as part of the directory path.