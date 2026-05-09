# All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation

## Abstract

Large Language Models (LLMs) have shown strong performance in low-resource offline translation. However, extending them to simultaneous speech-to-speech translation (Simul-S2ST) remains challenging due to the lack of causally aligned, speaker-preserving training data. Existing approaches often rely on fixed translation policies or confidence-based heuristics, which can lead to suboptimal translation quality and increased latency. We propose a causality-aware Simul-S2ST framework with a novel data generation pipeline that produces high-fidelity, causally aligned speech-to-speech segments while preserving speaker characteristics. The framework introduces three main components: (i) **FAST**, a factorized S2ST architecture; (ii) **CAP**, a causality-aware adaptive policy; and (iii) **CAAL**, a causality-aware latency metric. Experiments on CVSS Spanish, German, and French show that FAST improves translation quality by up to **+9.6 ASR-BLEU** over shared codec-based representations, while CAP improves performance by up to **+5.5 ASR-BLEU** and reduces latency by **26%** compared to a fixed policy.

**Paper:** []

## Repository Overview

It is organized into two main components:

1. A **data generation pipeline** for preparing causally aligned speech-to-speech training data.
2. A **training and evaluation recipe** for training, decoding, and evaluating simultaneous S2S-ST systems.

## Repository Structure

```text
FAST-CAP/
├── data_generation/   # Data preparation, alignment, TTS, and S2S shar generation
├── training/          # S2S-ST training, inference, evaluation, and streaming simulation
├── docs/              # Additional documentation
├── LICENSE
└── README.md
```
## Workflow Overview

The intended workflow is:
```
source speech and transcripts alignments
        ↓
text-to-text alignments
        ↓
target text-to-speech generation & alignments
        ↓
causal alignment conversion
        ↓
S2S shar generation
        ↓
model training and evaluation
```

The `data_generation/` directory produces the prepared training data used by the scripts in `training/`.

## Components

### 1. Data Generation Pipeline

The [`data_generation/`](data_generation/) directory contains scripts for preparing S2S-ST training data with causality-aware adaptive policy (CAP).

It includes:

- [`cvss2nemo/`](data_generation/cvss2nemo/): conversion utilities for CVSS-style data into NeMo-compatible formats.
- [`mfa/`](data_generation/mfa/): Montreal Forced Aligner-based speech-to-text alignment.
- [`tts_a2flow/`](data_generation/tts_a2flow/): A2Flow offline TTS.
- [`awesome_align_scripts/`](data_generation/awesome_align_scripts/): text-to-text translation alignment.
- [`mono_ali/`](data_generation/mono_ali/): conversion of raw alignments into monotonic alignments for streaming S2S modeling.
- [`extra_scripts/`](data_generation/extra_scripts/): additional helper scripts.

See [`data_generation/README.md`](data_generation/README.md) for detailed instructions.

### 2. Training, Evaluation, and Streaming

The [`training/`](training/) directory contains scripts for S2S-ST experiments.

It includes:

- [`s2s_exp/`](training/s2s_exp/): experiment scripts, configuration files, data preparation, training, and offline inference.
- [`s2s_metrics/`](training/s2s_metrics/): evaluation and analysis scripts, including BLEU, chrF, COMET, speaker similarity, ViSQOL, and DNSMOS.
- [`s2s_streaming/`](training/s2s_streaming/): streaming simulation, chunk-based inference, latency measurement, and long-form streaming evaluation.

See [`training/README.md`](training/README.md) for detailed instructions.

## Installation

Clone and install the modified versions of **Lhotse** and **NeMo** required by this pipeline.

### Install Lhotse

The data generation pipeline depends on a modified Lhotse branch with alignment support:

```bash
git clone https://github.com/AmirHussein96/lhotse.git
cd lhotse
git checkout alignments
pip install -e '.[dev]'
```    

### Install NeMo
```bash
git clone https://github.com/AmirHussein96/NeMo.git
cd NeMo
git checkout s2s_st2
pip install -e '.[all]'
```

Additional environment setup details are provided in the component-specific README files.

