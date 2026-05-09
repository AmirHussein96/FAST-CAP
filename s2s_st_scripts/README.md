# S2S_ST_scripts

This repository contains scripts and utilities for **FAST Speech-to-Speech Speech Translation (S2S-ST)** experiments, evaluation, and streaming simulation.

## Repository Structure

The repository is organized into three main directories:

### 1. `s2s_exp`
Contains scripts used to run **S2S-ST experiments**, along with their corresponding **configuration files**.  
These scripts cover data preparation, training, and inference workflows.

### 2. `s2s_metrics`
Includes all scripts related to **evaluation and analysis**, such as:
- Translation quality metrics (e.g., BLEU, chrF, COMET)
- Speaker similarity evaluation
- Audio and speech quality metrics (e.g., DNSMOS)

### 3. `s2s_streaming`
Contains scripts for **streaming and simultaneous S2S-ST experiments**, including:
- Streaming simulation and chunk-based processing
- Latency measurement and analysis
- Long-form streaming inference and evaluation