### Text to text alignment

You have two installation options for `awesome-align`:

- **Recommended (supports continued fine-tuning)**  
  Install from my fork, which adds support for resuming and continuing fine-tuning:  
  https://github.com/AmirHussein96/awesome-align

- **Official implementation**  
  Alternatively, install the official version by following the instructions in the original repository:  
  https://github.com/neulab/awesome-align


1. Prepate the paired text data:

    - Run the following script to extract and preprocess the parallel text data from nemo manifests:
        `sbatch run_prep_align.sh <LANG> <TGT_LANG>`

    - Below is a sample of the content of `.parallel` file. 

        ```
        A¿ dónde? A, a dónde te gustaría ir? ||| A, where? Which place would you like to go to?
        Que productos están en promoción, para no demorarme tanto y ya. ||| Which products are in promotion so I don't waste my time.
        ```

2. If you want to finetune the model combine the `*.parallel` generated data and create `train.parallel` and `dev.parallel`, either split the combined file or use different dataset for dev.

    - Use `bash split_train_dev.sh <parallel_path> <output_path>` 

3. Run finetuning for the Mbert on the parallel data

    - This step is recommended to get best text2text alignments.

    - This script expects `train.parallel` and `dev.parallel` in the `$OUTPUT_DIR`

        Once the data is ready, fine-tune the alignment model: `sbatch run_finetune_align.sh <LANG> <TGT_LANG>`

4. Generate alignments

    - After fine-tuning, run the alignment generation script: `sbatch generate_alignments.sh <LANG> <TGT_LANG> <model_name_or_path>`

