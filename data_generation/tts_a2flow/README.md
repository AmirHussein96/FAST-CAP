# a2flow-inference

### Build env

```
conda create -y -n a2flow python=3.10
conda activate a2flow
pip install unidecode
pip install ndjson
pip install phonemizer
```

### Install fairseq
https://github.com/facebookresearch/fairseq

To install Fairseq you need to:
- Downgrade pip `pip install pip==23.3.1`
- Install compatible Hydra + Omegaconf versions `pip install "omegaconf==2.0.6" "hydra-core==1.0.7"  --force-reinstall`
- Clone fairseq then: `pip install --editable ./`

### Install Lhotse (follow the instructions in README.md from the main folder)

#### Notes

For on-the-fly prompting during TTS generation, the pipeline requires:

- speech-to-text alignment files, such as CTM alignments;
- audio manifests;
- transcript manifests.

The data generation pipeline starts by generating speech-to-text alignments. These alignments can be produced using either MFA or NFA, as described in the corresponding component README files.


### (Optional) Update the dict used in a2flow
You can update the dictionary in the config path in `a-2-flow-inference-fork/configs` and then update `--text_processors` option in `a-2-flow-inference-fork/batched_inference.py`

### Extract ASR alignments (follow the instructions in the main data_generation/README.md)
This recipe supports on-the-fly prompt extraction. 
To use this feature, you need to provide the corresponding CTM file containing word-level alignments with the `--ctm` option.

### Download the pretrained models for TTS:

from SwiftStack `swift download playground -p amhussein/pretrained_models/a2flow` and replace path `pretrained_model=<path_to_pretrained_models>` in `submit_array.sh` 

### Generate the audio using TTS using array jobs
To run the TTS submit the array job `sbatch submit_array.sh`

For testing and debuging on single manifest follow: `batched_infer_test.sh`
