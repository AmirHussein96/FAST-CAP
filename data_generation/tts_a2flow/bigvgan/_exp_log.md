## parsing
```
submit_job -i -n bigvgan --email_mode never --duration 24 --partition cpu_long \
--image /lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:24.02-py3-v2.sqsh
```
```
PYTHONPATH=. python -m debugpy --listen 0.0.0.0:11111 --wait-for-client \
PYTHONPATH=. python \
parse_scripts/parse_adlr_audio.py \
--data_root /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AAM \
--output_path_root /lustre/fsw/portfolios/adlr/users/sanggill/filelists/adlr_audio_music_datasets_AAM \
--min_sampling_rate 8000 & \
PYTHONPATH=. python \
parse_scripts/parse_adlr_audio.py \
--data_root /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AAM \
--output_path_root /lustre/fsw/portfolios/adlr/users/sanggill/filelists/adlr_audio_music_datasets_AAM_44khz \
--min_sampling_rate 44100 & \
PYTHONPATH=. python \
parse_scripts/parse_adlr_audio.py \
--data_root /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AAM \
--output_path_root /lustre/fsw/portfolios/adlr/users/sanggill/filelists/adlr_audio_music_datasets_AAM_44khz_stereo \
--min_sampling_rate 44100 --require_stereo
```

## submit_job for interactive & long cpu-only processing
```
submit_job -i -n bigvgan --email_mode never --duration 24 --partition cpu_long \
--image /lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:24.02-py3-v2.sqsh
```

## auto-submit jobs (autoencoder exps using 24.02-py3-v2 image)
```
./submit_jobs_auto.sh \
experiments/experiments_ae.txt \
/lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:24.02-py3-v2.sqsh
```

## debug job
```
EXP_NAME=cqtloss_44k128b512h2048w_bs32_ss64k && \
python train_profile.py \
--config configs_poc/$EXP_NAME.json \
--dataset_config datasets/adlr_audio_commercial_44k.json \
--checkpoint_path tmp/$EXP_NAME \
--params batch_size=4 cqtd_filters=128
```

## adlr_audio_speech data debug job
```
EXP_NAME=v3loss_22k80b_bs128_ss16k && \
python -m debugpy --listen 0.0.0.0:11111 --wait-for-client \
train.py \
--debug True \
--config configs_poc/$EXP_NAME.json \
--dataset_config datasets/DEBUG_adlr_audio_speech.json \
--checkpoint_path tmp/$EXP_NAME
```
```
EXP_NAME=v3loss_22k80b_bs128_ss16k && \
python -m debugpy --listen 0.0.0.0:11111 --wait-for-client \
train.py \
--config configs_poc/$EXP_NAME.json \
--dataset_config datasets/libritts.json \
--checkpoint_path tmp/$EXP_NAME \
--validation_interval 1 \
--params batch_size=16 stereo=true
```

## inference (libritts pwc benchmark)
```
EXP_NAME=cqtlossf128_commercial_24k100b_bs32_ss64k_cgn500_notanh_nobias2 && \
CHECKPOINT_NAME=ema_0.999_g_03000000.pt && \
DATASET_NAME=LibriTTS-dev-benchmark && \
CUDA_VISIBLE_DEVICES=0 python inference.py \
--checkpoint_file /lustre/fsw/portfolios/adlr/users/sanggill/experiments/bigvgan/$EXP_NAME/$CHECKPOINT_NAME \
--input_wavs_dir /lustre/fsw/portfolios/adlr/users/sanggill/assets/$DATASET_NAME \
--output_dir /lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_bigvgan/$EXP_NAME/$CHECKPOINT_NAME/$DATASET_NAME &
```

```
EXP_NAME=bigvgan_24khz_100band && \
CHECKPOINT_NAME=g_05000000 && \
DATASET_NAME=reference_audio && \
python inference.py \
--checkpoint_file /lustre/fsw/portfolios/adlr/users/sanggill/external_models/BigVGAN_checkpoints/$EXP_NAME/$CHECKPOINT_NAME \
--input_wavs_dir /lustre/fsw/portfolios/adlr/users/sanggill/assets/$DATASET_NAME \
--output_dir /lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_bigvgan/$EXP_NAME/$CHECKPOINT_NAME/$DATASET_NAME &
```

```
EXP_NAME=dac_44khz_8kbps && \
python inference.py \
--checkpoint_file /lustre/fsw/portfolios/adlr/users/sanggill/experiments/bigvgan/cqtlossf128_commercial_44k128b512h2048w_bs32_ss64k_cgn500_notanh_nobias/ema_0.990_g_02500000.pt \
--dac_path /lustre/fsw/portfolios/adlr/users/sanggill/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth \
--input_wavs_dir /lustre/fsw/portfolios/adlr/users/sanggill/assets/reference_musdbhq \
--output_dir /lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_bigvgan/$EXP_NAME &
```

## autoencoder job
```
EXP_NAME=DEBUG_vae && \
rm -r tmp/$EXP_NAME && \
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs_vae/vae_oobleck_v3cqtloss_44k2048h_bs32_ss64k.json \
--dataset_config datasets/adlr_audio_everything_44k.json \
--checkpoint_path tmp/$EXP_NAME \
--params batch_size=4 update_d_every_n_steps=2 num_mels=128 perceptual_weighting=true use_sdstftloss=false warmup_step=1
```
```
EXP_NAME=DEBUG_vae && \
rm -r tmp/$EXP_NAME && \
python -m debugpy --listen 0.0.0.0:11111 --wait-for-client train.py \
--config configs_vae/vae_oobleck_v3cqtloss_44k2048h_bs32_ss64k.json \
--dataset_config datasets/adlr_audio_commercial_44k.json \
--checkpoint_path tmp/$EXP_NAME \
--validation_interval 100 \
--params update_d_every_n_steps=2 num_mels=128 num_workers=0 perceptual_weighting=true use_sdstftloss=false
```

## autoencoder recon inference
EXP_NAME=vae_wavenet384_all_cqtlossf128_44k128b512h2048w_bs32_ss64k_vd128_kl1e-6_cgn2000_noaatanhbias && \
CHECKPOINT_NAME=ema_0.990_g_01700000.pt && \
python inference.py \
--checkpoint_file /lustre/fsw/portfolios/adlr/users/sanggill/experiments/ae_bigvgan/$EXP_NAME/$CHECKPOINT_NAME \
--input_wavs_dir /lustre/fsw/portfolios/adlr/users/sanggill/projects/playground/assets/reference_audio \
--output_dir /lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_bigvgan/$EXP_NAME/$CHECKPOINT_NAME

## evaluate
python -m debugpy --listen 0.0.0.0:11111 --wait-for-client evaluate.py \
python evaluate.py \
--checkpoint_path /lustre/fsw/portfolios/adlr/users/sanggill/experiments/bigvgan/all_v3loss_24k100b_bs128_ss16k_cgn500 \
--dataset_config datasets/libritts_24k.json \
--ema_decay 0.999 --save_ema

## rsync samples to desktop for eval
rsync -au -P /lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_bigvgan sanggill@10.110.37.24:/home/sanggill/temp

## evaluate_bigvsan (for PwC benchmark) on desktop
EXP_NAME=cqtlossf128_commercial_24k100b_bs32_ss64k_cgn500_notanh_nobias2 && \
CHECKPOINT_NAME=ema_0.990_g_03000000.pt && \
DATASET_NAME=LibriTTS-dev-benchmark && \
CUDA_VISIBLE_DEVICES=1 python evaluate_bigvsan.py \
/home/sanggill/temp/sample_bigvgan/$EXP_NAME/$CHECKPOINT_NAME/$DATASET_NAME/real \
/home/sanggill/temp/sample_bigvgan/$EXP_NAME/$CHECKPOINT_NAME/$DATASET_NAME/generated &

## evaluate audio dir
AUDIO_DIR=/lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_stable_audio/stable-audio-open-1.0-vae/model_unwrap/LibriTTS-dev-benchmark && \
CUDA_VISIBLE_DEVICES=0 python evaluate_audio_dir.py \
--audio_dir $AUDIO_DIR \
--skip_mcd_v1



## export
python export_to_torchscript.py \
--checkpoint_file /lustre/fsw/portfolios/adlr/users/sanggill/experiments/ae_bigvgan/vae_wavenet384_all_v3loss_22k80b_vd128_kl1e-6_cgn1000/ema_0.990_g_03250000.pt