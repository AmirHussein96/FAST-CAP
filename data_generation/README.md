# S2S Data Generation Pipeline

This repository contains the data preparation pipeline for generating causality-aware speech-to-speech (S2S) training shards.

The pipeline consists of four main stages:

1. speech-to-text alignment,
2. target speech generation using TTS,
3. source-to-target text alignment,
4. monotonic alignment and S2S shard generation.

---

## Step 1: Generate Speech-to-Text Alignments

Speech-to-text alignments can be generated using either **MFA** or **NFA**. We recommend using **MFA**, as it generally provides more accurate word boundary alignments.

- **MFA alignments:** see the `README.md` file in the `mfa/` directory.
- **NFA alignments:** see the `README.md` file in the `nfa/` directory.

---

## Step 2: Generate Target Speech with TTS

Target speech is generated using A2Flow TTS.

- **A2Flow TTS:** see the `README.md` file in the `tts_a2flow/` directory.

---

## Step 3: Generate Source-to-Target Text Alignments

Source-to-target text alignments are generated using Awesome-Align.

- See the `README.md` file in the `awesome_align_scripts/` directory.

---

## Step 4: Generate S2S Shards

The final step converts raw source–target alignments into monotonic alignments suitable for streaming S2S modeling.

This stage produces causality-aware training shards that can be used for simultaneous speech-to-speech translation experiments.

- See the scripts and command examples in the `mono_ali/` directory.

---

## Regenerated CVSS-T Data Resources

Generated CVSS-T resources are available on Hugging Face:

- A2Flow-generated speech and alignments:  
  `https://huggingface.co/datasets/AmirHussein/Causality-Aware-CVSS/tree/main/generated_resources`

- Sample causality-aware adaptive policy shards generated using `mono_ali/`:  
  `https://huggingface.co/datasets/AmirHussein/Causality-Aware-CVSS/tree/main/cap_prep_sample/cvss_dev_es`

