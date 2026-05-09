# **S2S-ST Setup & Usage Guide**

This guide provides step-by-step instructions to set up, train, and run inference for the  
**Speech-to-Speech Speech Translation (S2S-ST)** pipeline using **NVIDIA NeMo**.

---

## 1. Prerequisites

### Noise Data

- For noise augmentation, you can use any background-noise dataset. One commonly used option is **MUSAN**: 

```text
https://www.openslr.org/17/
```


## 2. Download Pretrained Models

Download the pretrained ASR, TTS, and codec models from the FAST-CAP Hugging Face repository:

```text
https://huggingface.co/AmirHussein/FAST-CAP/tree/main/models/asr
https://huggingface.co/AmirHussein/FAST-CAP/tree/main/models/tts
https://huggingface.co/AmirHussein/FAST-CAP/tree/main/models/codec
```

These components are used to initialize different parts of the S2S-ST pipeline.

---


## 3. Download CVSS-T Data Resources

Download the prepared CVSS-T generated resources from Hugging Face:

```text
https://huggingface.co/datasets/AmirHussein/Causality-Aware-CVSS/tree/main/generated_resources
```
These resources include:

- A2Flow-generated English target speech with improved speaker preservation;
- source-side speech-to-text alignments;
- target-side speech-to-text alignments;
- source-to-target text alignments.

You also need the original CVSS-T source data:

```text
https://github.com/google-research-datasets/cvss
```

If you want to regenerate the causality-aware translation data from scratch, follow the data generation pipeline in:

```text
FAST-CAP/data_generation
```

For more details, see the README files inside the corresponding subdirectories of `FAST-CAP/data_generation`.

--- 


## 4. Run Training Experiments

Each experiment directory contains the scripts and configuration files required to launch S2S-ST training.

### Training Script

To launch the multilingual causality-aware S2S-ST training experiment, run:

```bash
sbatch s2s_exp/train_qwen_1.5b_encoder_70_0_st_mfa_multiling_causality_aware.sh
```
This script launches training using the Qwen-based S2S-ST model with MFA-based alignments and causality-aware multilingual CVSS-T data.

### Configuration Files

The experiment configuration files are located in `s2s_exp/configs`.

Before launching training, update the configuration file with the correct paths for:

- pretrained ASR model;
- pretrained audio codec;
- prepared CVSS-T training and validation shards;
- noise data, if noise augmentation is enabled;
- output directory for checkpoints and logs.

### Pretrained S2S Checkpoint

A pretrained multilingual FAST-CAP S2S checkpoint is available at:

https://huggingface.co/AmirHussein/FAST-CAP/tree/main/models/s2s_fast_cap_multiling

---

## 5. Run Inference

For batch inference, use:

``` bash
sbatch inference_qwen_1.5b_encoder_70_0_st.sh
```

This script runs batch decoding using a trained S2S-ST checkpoint.

---

## 6. Notes and Prerequisites for Inference

Before running inference, verify the following requirements.

### Checkpoint Filename

Ensure that the checkpoint filename does **not** contain the `=` character. This character can cause parsing or checkpoint-loading failures in some scripts.

### Training Script Compatibility

Replace the default NeMo training script with the exact version used during training to ensure checkpoint compatibility:

```python
Nemo/examples/speechlm2/train.py
```

This is important because the model definition, checkpoint structure, or dataloader behavior may differ from the default NeMo version.

### Model Configuration Updates

In the inference configuration file `exp_config.yaml`, set the correct paths for all pretrained and trained components:

- `pretrained_audio_codec: /path/to/audio_codec`
- `pretrained_asr: /path/to/asr_model`
- `pretrained_s2s_model: /path/to/s2s_checkpoint`

Make sure these paths match the models downloaded from the FAST-CAP Hugging Face repository or the checkpoints generated during training.
