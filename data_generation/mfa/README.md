# S2S_DataGeneration_pipeline

## Generate Text to Speech Alignments
1. Alignments with MFA: 
Build MFA env
```
conda create -y -n mfa python=3.9
conda activate mfa
conda config --add channels conda-forge
pip install seaborn
conda install -y montreal-forced-aligner
pip install joblib==1.2.0
pip install unidecode
pip install ndjson

#get the pretrained MFA models and prepared dictionaries
mfa model download dictionary english_mfa --github_token <token>
mfa model download acoustic english_mfa --github_token <token>
mfa model download dictionary spanish_mfa --github_token <token>
mfa model download acoustic spanish_mfa --github_token <token>
mfa model download dictionary german_mfa --github_token <token>s
mfa model download acoustic german_mfa --github_token <token>
mfa model download dictionary french_mfa --github_token <token>
mfa model download acoustic french_mfa --github_token <token>
```

### Untar the audio 

```
sbatch extract_array.sh
```



### Run MFA align  
```
sbatch run_mfa.sh <SRC_LANG>   <TGT_LANG>  <TTS>    # <SRC_LANG>: es, en, fr, de; <TGT_LANG>: es, en, de; <TTS>: if the input is from TTS folder, use MFA model of target language
```

### (Optional) validate the dir and check for oovs#
After preparing the text and wav files in same folder you can run this to validate the directory

```
mfa validate --single_speaker <path_to_dir> spanish_mfa
```