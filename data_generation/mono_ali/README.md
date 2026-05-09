## Preparing Lhotse SHARs for S2S-ST Training

This step combines all generated inforamtion (audio, text, and alignments) into Lhotse SHAR archives for training and evaluation of the S2S-ST models.

#### Training Set Generation

- **Fixed Source Chunking**  

  Uses fixed-length source audio chunks, independent of alignment information.

    ```bash
    sbatch combine_info_concat_v.sh
    ```

- **Dynamic Target-Aligned Chunking**  

  Constructs causality-aware adaptive chunking.
    
    ```bash
    sbatch combine_info_concat_v_target_aligned.sh
    ```  

#### Evaluation Set Generation

- Evaluation does not require alignment information.  The evaluation manifests include only source speech and the corresponding target text translations.
    
    ```bash
    sbatch combine_info_concat_v_eval.sh
    ```