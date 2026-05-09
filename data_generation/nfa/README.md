
## Generate Text to Speech Alignments using NFA

Download the ASR models for NFA
Spanish: https://huggingface.co/nvidia/stt_es_conformer_ctc_large 
German: https://huggingface.co/nvidia/stt_de_conformer_ctc_large 


### Speech to text alignment using NFA
First clone my Nemo repo (https://github.com/AmirHussein96/NeMo.git) and branch out to `nfa_lhotse_tar`
Install Nemo
 ```
 To run array job:
 sbatch run_nemo_forced_aligner_local_es_array.sh
```
Note: The ASR model will be selected based on the language abbreviation in the job name. For example, jobs with *_es will automatically select the Spanish model.


To test NFA on a single file:
```
# MODEL_DIR=/lustre/fsw/portfolios/edgeai/users/amhussein/pretrained_models

# manifest_number=3
# manifest_file=manifest_${manifest_number}.json
# tar_path=audio_${manifest_number}.tar
python NeMo/tools/nemo_forced_aligner/align.py \
        batch_size=64 \
        model_path=pretrained_models/Parakeet-Hybrid-XL-unified-0.6b_spe1024_en-US_1.0.nemo \
        manifest_filepath="$manifest_file" \
        save_output_file_formats=\[ctm\] \
        output_dir=${manifest_number} \
        load_lhotse_tarred=True \
        combine_ctms=True \
        tar_path=$tar_path \
        output_timestep_duration=0.08 \
        time_id=True
```

