# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
os.system("ulimit -c 0") # disable core dump in case of unknown errors on cluster job
import time
import argparse
import json
from pesq import pesq
from tqdm import tqdm
import auraloss
import random
from env import AttrDict, build_env, update_params
from shutil import copyfile
import dac

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import torchaudio as ta

from meldataset import MelDataset, mel_spectrogram, load_data, MAX_WAV_VALUE
from models import BigVGAN, LatentAutoEncoder, apply_generator_forward
from discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator, MultiResolutionDiscriminatorDAC, MultiScaleSubbandCQTDiscriminator, CombinedDiscriminator
from loss import MelSpectrogramLoss, feature_loss, generator_loss, discriminator_loss, kl_loss
from utils import plot_spectrogram, plot_spectrogram_clipped, scan_checkpoint, load_checkpoint, save_checkpoint, save_audio
import pytorch_warmup as warmup

torch.backends.cudnn.benchmark = False
TOTAL_RUN_TIME = 13800 # 3 hours 50 minutes (10 min before 4h cutoff)
START_TIME = time.time()

def load_generator(model_type, h, device):
    print(f"model_type is {model_type}")
    if model_type == "vocoder":
        # define BigVGAN generator
        generator = BigVGAN(h).to(device)
        print("Vocoder params: {}".format(sum(p.numel() for p in generator.parameters())))
    elif model_type in ["autoencoder", "vae"]:
        generator = LatentAutoEncoder(h).to(device)
        print("Autoencoder params: {}".format(sum(p.numel() for p in generator.parameters())))
        print("Encoder params: {}".format(sum(p.numel() for p in generator.encoder.parameters())))
        print("Decoder params: {}".format(sum(p.numel() for p in generator.decoder.parameters())))
    return generator

def load_discriminator(h, device):
    # define discriminators. MPD is used by default
    # to keep preivous code structure, mpd is not wrapped to CombinedDiscriminator (if possibly used)
    mpd = MultiPeriodDiscriminator(h).to(device)

    # define additional discriminators. BigVGAN-v2 uses DAC's MRD as default
    # NEW: if use_dac_mrd_instead_of_original=True, it switches UnitNet MRD to DAC MRD
    if getattr(h, "use_dac_mrd_instead_of_original", False):
        print("WARNING: changing MRD to DAC's MRD!")
        mrd = MultiResolutionDiscriminatorDAC(h).to(device)
        # NEW: add cqtd to MRD
        if getattr(h, "add_cqtd_to_mrd", False):
            print("WARNING: adding cqtd on top of mrd (i.e., DAC MRD + CQTD into one mrd module!)")
            cqtd = MultiScaleSubbandCQTDiscriminator(h).to(device)
            mrd = CombinedDiscriminator([mrd, cqtd])
    elif getattr(h, "use_cqtd_instead_of_mrd", False): # or ist switches MRD to CQTD
        assert not getattr(h, "add_cqtd_to_mrd", False), "only use one option: use_cqtd_instead_of_mrd OR add_cqtd_to_mrd"
        print("WARNING: changing MRD to CQTD!")
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:
        assert not getattr(h, "add_cqtd_to_mrd", False), "original MRD + cqtd is not supported"
        print("INFO: using original MRD from UnivNet (fallback discrimiantor of BigVGAN-v1)")
        mrd = MultiResolutionDiscriminator(h).to(device)
    
    return mpd, mrd

#####################################################
# validation loop
# "mode" parameter is automatically defined as (seen or unseen)_(name of the dataset)
# if the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors
#####################################################
def validate(rank, generator, a, h, loader, steps, device, sw, mode="seen"):
    assert rank == 0, "validate should only run on rank=0"
    generator.eval()
    torch.cuda.empty_cache()

    val_err_tot = 0
    val_pesq_tot = 0
    val_mrstft_tot = 0

    # modules for evaluation metrics
    pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).cuda()
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

    if a.save_audio: # also save audio to disk if --save_audio is set to True
        os.makedirs(os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode)), exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps)), exist_ok=True)

    with torch.no_grad():
        print("step {} {} speaker validation...".format(steps, mode))

        # loop over validation set and compute metrics
        for j, batch in tqdm(enumerate(loader)):
            x_linear, x_mel, y, audiopath, y_mel = batch["linear_spec"], batch["mel"], batch["audio"], batch["audiopath"], batch["mel_loss"]
            # choose input representation
            if getattr(h, "use_wav_as_input", False):
                x = y.clone()
            elif getattr(h, "use_linear_spec_as_input", False):
                x = x_linear
            else:
                x = x_mel
            y = y.to(device)
            
            # apply model forward. encoder_out and latent are avilable only for autoencoder (for vocoder, both are None)
            return_dict = apply_generator_forward(h.model_type, generator, x.to(device))
            y_g_hat = return_dict["decoder_out"]
                
            y_mel = y_mel.to(device, non_blocking=True)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
            
            y_mono = torch.mean(y, dim=1) # [B, T]
            y_g_hat_mono = torch.mean(y_g_hat, dim=1) # [B, T]
            # PESQ calculation. only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
            if not "nonspeech" in mode: # skips if the name of dataset (in mode string) contains "nonspeech"
                # resample to 16000 for pesq
                y_16k = pesq_resampler(y_mono)
                y_g_hat_16k = pesq_resampler(y_g_hat_mono)
                y_int_16k = (y_16k * MAX_WAV_VALUE).short().cpu().numpy()
                y_g_hat_int_16k = (y_g_hat_16k * MAX_WAV_VALUE).short().cpu().numpy()
                val_pesq_tot += pesq(16000, y_int_16k[0], y_g_hat_int_16k[0], 'wb')

            # MRSTFT calculation
            val_mrstft_tot += loss_mrstft(y_g_hat, y).item()

            # log audio and figures to Tensorboard
            if j % a.eval_subsample == 0:  # subsample every nth from validation set
                if steps >= 0:
                    sw.add_audio('gt_{}/y_{}'.format(mode, j), y_mono, steps, h.sampling_rate)
                    if a.save_audio: # also save audio to disk if --save_audio is set to True
                        save_audio(y[0], os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode), '{:04d}.wav'.format(j)), h.sampling_rate)
                    sw.add_figure('gt_{}/y_spec_{}'.format(mode, j), plot_spectrogram(x[0]), steps)
                sw.add_audio('generated_{}/y_hat_{}'.format(mode, j), y_g_hat_mono, steps, h.sampling_rate)
                if a.save_audio: # also save audio to disk if --save_audio is set to True
                    save_audio(y_g_hat[0].permute(0, 1), os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps), '{:04d}.wav'.format(j)), h.sampling_rate)
                # spectrogram of synthesized audio
                y_hat_spec = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                sw.add_figure('generated_{}/y_hat_spec_{}'.format(mode, j), plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)
                # visualization of spectrogram difference between GT and synthesized audio
                # difference higher than 1 is clipped for better visualization
                spec_delta = torch.clamp(torch.abs(y_mel.cpu()[0] - y_hat_spec.squeeze(0).cpu()), min=1e-6, max=1.)
                sw.add_figure('delta_dclip1_{}/spec_{}'.format(mode, j), plot_spectrogram_clipped(spec_delta.numpy(), clip_max=1.), steps)
                # plot latent for autoencoders
                if "latent" in return_dict and return_dict["latent"] is not None:
                    latent = return_dict["latent"][0].cpu().numpy()
                    sw.add_figure('latent_{}/latent_{}'.format(mode, j), plot_spectrogram(latent), steps)
                # also plot mu and logvar for vae
                if "mu" in return_dict and return_dict["mu"] is not None:
                    mu = return_dict["mu"][0].cpu().numpy()
                    sw.add_figure('vae_mu_{}/vae_mu_{}'.format(mode, j), plot_spectrogram(mu), steps)
                if "logvar" in return_dict and return_dict["logvar"] is not None:
                    logvar = return_dict["logvar"][0].cpu().numpy()
                    sw.add_figure('vae_logvar_{}/vae_logvar_{}'.format(mode, j), plot_spectrogram(logvar), steps)

        val_err = val_err_tot / (j + 1)
        val_pesq = val_pesq_tot / (j + 1)
        val_mrstft = val_mrstft_tot / (j + 1)
        # log evaluation metrics to Tensorboard
        sw.add_scalar("validation_{}/mel_spec_error".format(mode), val_err, steps)
        sw.add_scalar("validation_{}/pesq".format(mode), val_pesq, steps)
        sw.add_scalar("validation_{}/mrstft".format(mode), val_mrstft, steps)

    generator.train()

    
def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank
        )

    # set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))
    
    # load model. can be stand-alone vocoder or latent autoencoder
    generator = load_generator(h.model_type, h, device)

    # load discriminators: mpd and mrd. mrd can be either original univnet or improved alternatives
    mpd, mrd = load_discriminator(h, device)
    
    # NEW: if use_dac_melloss_instead_of_l1=True, it switches HiFi-GAN L1 mel loss to DAC multi-scale mel loss
    if getattr(h, "use_dac_melloss_instead_of_l1", False):
        print("WARNING: changing Mel l1 loss to DAC multi-scale version!")
        fn_mel_loss_dac = MelSpectrogramLoss(
            sampling_rate=h.sampling_rate,
            perceptual_weighting=getattr(h, "perceptual_weighting", False)
        ) # NOTE: accepts waveform as input
    # NEW: if use_mrstft_instead_of_l1=True, it switches HiFi-GAN L1 mel loss to MRSTFT from auraloss
    # https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/training/autoencoders.py#L128
    elif getattr(h, "use_mrstft_instead_of_l1", False):
        print("WARNING: changing Mel l1 loss to MRSTFT version!")
        fn_mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=h.sampling_rate, **h.mrstft_config)
    else:
        print("INFO: using original Mel l1 loss from HiFi-GAN/BigVGAN")
        fn_mel_loss_hfg = F.l1_loss
    
    # NEW: stereo specific loss (sum and difference)
    if getattr(h, "stereo", False) and getattr(h, "use_sdstft_loss", False):
        print("WARNING: using stereo mel loss (sum-and-difference + existing mono spectral loss)!")
        fn_sdstft_loss = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=h.sampling_rate, **h.mrstft_config)
    
    #####################################################
    # define loss_type for discriminator_loss and generator_loss
    # defaults to "l2" (LSGAN, as in original hifi-gan/bigvgan). can select "hinge" (used by encodec, vocos, etc)
    #####################################################
    loss_type = getattr(h, "loss_type", "l2")
    if loss_type != "l2":
        print(f"WARNING: using {loss_type} for discriminator_loss and generator_loss instead of default (l2)!")

    #####################################################
    # loading checkpoints
    #####################################################
    # create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print("Discriminator mpd params: {}".format(sum(p.numel() for p in mpd.parameters())))
        print("Discriminator mrd params: {}".format(sum(p.numel() for p in mrd.parameters())))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_', idx_to_load=-1)
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_', idx_to_load=-1)

    # load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        try:
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
        except RuntimeError: # last checkpoint corrupted
            print("WARNING: last checkpoints are corrupted. trying to load second last..")
            cp_g = scan_checkpoint(a.checkpoint_path, 'g_', idx_to_load=-2)
            cp_do = scan_checkpoint(a.checkpoint_path, 'do_', idx_to_load=-2)
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    #####################################################
    # initialize DDP, optimizers, and schedulers
    #####################################################
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
        weight_decay=getattr(h, "weight_decay", 0.01)
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
        weight_decay=getattr(h, "weight_decay", 0.01)
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)
    
    WARMUP_PHASE = False
    if steps == 0:
        warmup_g = warmup.LinearWarmup(optim_g, warmup_period=h.warmup_step)
        warmup_d = warmup.LinearWarmup(optim_d, warmup_period=h.warmup_step)
        WARMUP_PHASE = True

    if state_dict_do is not None:
        scheduler_g.load_state_dict(state_dict_do['scheduler_g'])
        scheduler_d.load_state_dict(state_dict_do['scheduler_d'])
    
    #####################################################
    # define training and validation datasets
    # unseen_validation_filelist will contain sample filepaths outside the seen training & validation dataset
    # example: trained on LibriTTS, validate on VCTK
    # training_filelist, validation_filelist, list_unseen_validation_filelist = get_dataset_filelist(a)
    #####################################################
    ignore_keys = ['training_files', 'validation_files']
    training_filelist = load_data(
        h.data["training_files"],
        pre_shuffle=True,
        **dict((k, v) for k, v in h.data.items() if k not in ignore_keys)
        )
    print("INFO: shuffling training_filelist before training")
    random.shuffle(training_filelist)
    
    validation_filelist = load_data(
        h.data["validation_files"],
        pre_shuffle=False,
        **dict((k, v) for k, v in h.data.items() if k not in ignore_keys)
        )
    list_unseen_validation_filelist = [] # skip for now

    trainset = MelDataset(
        training_filelist, h,
        split=True, shuffle=False if h.num_gpus > 1 else True, n_cache_reuse=0, device=device,
        fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=True, debug=a.debug
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=False,
        sampler=train_sampler, batch_size=h.batch_size, pin_memory=True, drop_last=True
    )

    if rank == 0:
        validset = MelDataset(
            validation_filelist, h, 
            split=False, shuffle=False, n_cache_reuse=0, device=device,
            fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=True, debug=a.debug
        )
        validation_loader = DataLoader(
            validset, num_workers=1, shuffle=False,
            sampler=None, batch_size=1, pin_memory=True, drop_last=True
        )

        list_unseen_validset = []
        list_unseen_validation_loader = []
        for i in range(len(list_unseen_validation_filelist)):
            unseen_validset = MelDataset(
                list_unseen_validation_filelist[i], h,
                split=False, shuffle=False, n_cache_reuse=0, device=device,
                fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=False, debug=a.debug
            )
            unseen_validation_loader = DataLoader(
                unseen_validset, num_workers=1, shuffle=False,
                sampler=None, batch_size=1, pin_memory=True, drop_last=True
            )
            list_unseen_validset.append(unseen_validset)
            list_unseen_validation_loader.append(unseen_validation_loader)

        # Tensorboard logger
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        if a.save_audio: # also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, 'samples'), exist_ok=True)
    
    def save_checkpoint_during_training(steps, use_last_as_suffix=True):
        if use_last_as_suffix:
            checkpoint_path = "{}/g_last".format(a.checkpoint_path)
            if os.path.exists(checkpoint_path):
                checkpoint_path_backup = checkpoint_path.replace("g_last", "backup_g_last")
                copyfile(checkpoint_path, checkpoint_path_backup)            
        else:
            checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
        save_checkpoint(checkpoint_path,
                        {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
        
        if use_last_as_suffix:
            checkpoint_path = "{}/do_last".format(a.checkpoint_path)
            if os.path.exists(checkpoint_path):
                checkpoint_path_backup = checkpoint_path.replace("do_last", "backup_do_last")
                copyfile(checkpoint_path, checkpoint_path_backup)       
        else:
            checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
        save_checkpoint(checkpoint_path, 
                        {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                        'mrd': (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'scheduler_g': scheduler_g.state_dict(),
                        'scheduler_d': scheduler_d.state_dict(),
                        'steps': steps,
                        'epoch': epoch})

    # if the checkpoint is loaded, start with validation loop
    # if steps >= 0 and rank == 0 and not a.debug:
    #     if not a.skip_seen:
    #         validate(rank, generator, codec_model, a, h, validation_loader, steps, device, sw,
    #                  mode="seen_{}".format(train_loader.dataset.name))
    #     for i in range(len(list_unseen_validation_loader)):
    #         validate(rank, generator, codec_model, a, h, list_unseen_validation_loader[i], steps, device, sw,
    #                  mode="unseen_{}".format(list_unseen_validation_loader[i].dataset.name))
    # exit the script if --evaluate is set to True
    if a.evaluate:
        exit()
        
    #####################################################
    # start of training logic
    #####################################################
    print("INFO: now starting main training loop!")
    generator.train()
    mpd.train()
    mrd.train()
    if a.freeze_step > 0:
        print("WARNING: skipping D training and regression loss only for G for the first {} steps".format(a.freeze_step))
        
    # NEW: update D for every N generator update steps.
    # original hifi-gan/bigvgan updates D & G every steps simultaneously, but it seems to cause instability for wav-in wav-out autoencoder
    # use similar approach to stable audio tools
    update_d_every_n_steps = getattr(h, "update_d_every_n_steps", 1)
    if update_d_every_n_steps > 1:
        print(f"WARNING: update_d_every_n_steps set to {update_d_every_n_steps}!")
    
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        if a.debug: # dataset sanity check
            print("INFO: debug mode is on. Checking if there's no error on a full epoch of train_loader...")
            tqdm_object = tqdm(train_loader, disable=rank != 0)
            for i, batch in enumerate(tqdm_object):
                x_linear, x_mel, y, audiopath, y_mel = batch["linear_spec"], batch["mel"], batch["audio"], batch["audiopath"], batch["mel_loss"]
                # choose input representation
                if getattr(h, "use_wav_as_input", False):
                    x = y.clone()
                elif getattr(h, "use_linear_spec_as_input", False):
                    x = x_linear
                else:
                    x = x_mel
                steps += 1
                # Update the description for each iteration
                lr = optim_g.param_groups[0]['lr']
                # Update the tqdm progress bar postfix with the current step and learning rate
                tqdm_object.set_postfix({'Step': steps, 'lr': f'{lr:4.7f}'}, refresh=True)
                scheduler_g.step()
                scheduler_d.step()
                continue
            print("INFO: full epoch of train_loader has passed. No issue on data! Exising...")
            exit()
        
        #####################################################
        # main training loop
        #####################################################
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
                if start_b - START_TIME > TOTAL_RUN_TIME:
                    save_checkpoint_during_training(steps, use_last_as_suffix=True)
                    print("INFO: TOTAL_RUN_TIME {} reached. Exiting the script gracefully. See you in the next run!".format(TOTAL_RUN_TIME))
                    exit()
                    
            x_linear, x_mel, y, audiopath, y_mel = batch["linear_spec"], batch["mel"], batch["audio"], batch["audiopath"], batch["mel_loss"]
            # choose input representation
            if getattr(h, "use_wav_as_input", False):
                x = y.clone()
            elif getattr(h, "use_linear_spec_as_input", False):
                x = x_linear
            else:
                x = x_mel
            
            x = x.to(device)
            y = y.to(device)
            y_mel = y_mel.to(device)
            # y = y.unsqueeze(1)
            
            #####################################################
            # generator forward pass
            #####################################################
            # apply model forward. encoder_out and latent are avilable only for autoencoder (for vocoder, both are None)
            return_dict = apply_generator_forward(h.model_type, generator, x)
            y_g_hat = return_dict["decoder_out"]
            
            #####################################################s
            # discriminator forward, loss & backward pass
            #####################################################
            if steps % update_d_every_n_steps == 0:
                optim_d.zero_grad()
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g, loss_type=loss_type)
                # MRD
                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g, loss_type=loss_type)
                loss_disc_all = loss_disc_s + loss_disc_f
                # whether to freeze D for initial training steps
                clip_grad_norm_d = getattr(h, "clip_grad_norm_d", h.clip_grad_norm) # override if exists
                if steps >= a.freeze_step:
                    loss_disc_all.backward()
                    grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), clip_grad_norm_d)
                    grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), clip_grad_norm_d)
                    optim_d.step()
                else:
                    grad_norm_mpd = 0.
                    grad_norm_mrd = 0.

            #####################################################
            # generator loss & backward pass
            #####################################################
            optim_g.zero_grad()
            # whether to normalize volume during melloss calculation
            if getattr(h, "melloss_normalize_volume", False):
                # Remove DC offset
                y_g_hat_for_melloss = y_g_hat - y_g_hat.mean(dim=-1, keepdims=True)
                y_for_melloss = y - y.mean(dim=-1, keepdims=True)
                # Peak normalize the volume of input audio
                y_g_hat_for_melloss = 0.8 * y_g_hat_for_melloss / (y_g_hat_for_melloss.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
                y_for_melloss = 0.8 * y_for_melloss / (y_for_melloss.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
            else:
                y_g_hat_for_melloss = y_g_hat
                y_for_melloss = y
            
            # Mel-Spectrogram Loss: DAC's multi-scale version
            if getattr(h, "use_dac_melloss_instead_of_l1", False): # uses waveform as input & mel fn from dac
                loss_mel = fn_mel_loss_dac(y_g_hat_for_melloss, y_for_melloss) * h.lambda_melloss
            # MRSTFT loss with stereo version from stable audio 2.0. fn(x=x_g_hat, y=y)
            elif getattr(h, "use_mrstft_instead_of_l1", False):
                loss_mel = fn_mrstft_loss(y_g_hat_for_melloss, y_for_melloss) * h.lambda_melloss
            # original single-scale mel l1 loss form HiFi-GAN/BigVGAN. uses previous <y_mel, y_g_hat_mel> for loss
            else:
                y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                loss_mel = fn_mel_loss_hfg(y_mel, y_g_hat_mel) * h.lambda_melloss
            # Add sum and difference loss for stereo setup if specified
            if getattr(h, "stereo", False) and getattr(h, "use_sdstft_loss", False): 
                assert y_for_melloss.shape[1] == 2 and y_g_hat_for_melloss.shape[1] == 2, \
                    f"model is stereo but y and y_g_hat is not! got y.shape {y.shape} y_g_hat.shape {y_g_hat.shape}"
                loss_sdstft = torch.clamp(fn_sdstft_loss(y_g_hat_for_melloss, y_for_melloss) * h.lambda_sdstftloss, max=10) # empirical clamping to prevent divergence
                if not torch.isnan(loss_sdstft):
                    loss_mel = loss_mel + loss_sdstft
            
            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g, loss_type=loss_type)
            
            # MRD loss: either original UnivNet's MRD or DAC's improved MRD/MCBD
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g, loss_type=loss_type)

            if steps >= a.freeze_step:
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                if h.model_type == "vae":
                    mu, logvar = return_dict["mu"], return_dict["logvar"]
                    assert mu is not None and logvar is not None and hasattr(h, "lambda_klloss")
                    loss_kl = kl_loss(mu, logvar) * h.lambda_klloss
                    loss_gen_all = loss_gen_all + loss_kl
            else:
                loss_gen_all = loss_mel
                
            loss_gen_all.backward()
            clip_grad_norm_g = getattr(h, "clip_grad_norm_g", h.clip_grad_norm) # override if exists
            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad_norm_g)
            optim_g.step()

            #####################################################
            # logging on rank 0
            #####################################################
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()                        
                    base_message = (
                        f'Steps : {steps:d}, '
                        f'Gen Loss Total : {loss_gen_all:4.3f}, '
                        f'Mel Error : {mel_error:4.3f}, '
                        f's/b : {time.time() - start_b:4.3f}, '
                        f'lr : {optim_g.param_groups[0]["lr"]:4.7f}'
                    )
                    if h.model_type == "vae":
                        base_message += f', VAE-KL : {loss_kl.item():4.3f}'
                    print(base_message)

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    save_checkpoint_during_training(steps, use_last_as_suffix=False)
                    
                if steps % 1000 == 0 and steps != 0:
                    # 1000-step wise last checkpointing to _last for auto-resume
                    save_checkpoint_during_training(steps, use_last_as_suffix=True)

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                    sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                    sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                    sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                    sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                    sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                    sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                    sw.add_scalar("training/gen_loss_mel", loss_mel.item(), steps)
                    if getattr(h, "stereo", False) and getattr(h, "use_sdstft_loss", False):
                        sw.add_scalar("training/gen_loss_sdstft", loss_sdstft.item(), steps)
                    sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                    sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                    sw.add_scalar("training/learning_rate_d", optim_d.param_groups[0]['lr'], steps)
                    sw.add_scalar("training/learning_rate_g", optim_g.param_groups[0]['lr'], steps)
                    sw.add_scalar("training/epoch", epoch+1, steps)
                    if h.model_type == "vae":
                        sw.add_scalar("training/kl_loss", loss_kl.item(), steps)

                # validation
                if steps % a.validation_interval == 0:
                    # plot training input x so far used
                    for i_x in range(x.shape[0]):
                        sw.add_figure('training_input/x_{}'.format(i_x), plot_spectrogram(x[i_x].cpu()), steps)
                        sw.add_audio('training_input/y_{}'.format(i_x), y[i_x][0], steps, h.sampling_rate)

                    # seen and unseen speakers validation loops
                    if not a.debug and steps != 0:
                        validate(rank, generator, a, h, validation_loader, steps, device, sw, mode="seen_{}".format(train_loader.dataset.name))
                        for i in range(len(list_unseen_validation_loader)):
                            validate(rank, generator, a, h, list_unseen_validation_loader[i], steps, device, sw, mode="unseen_{}".format(list_unseen_validation_loader[i].dataset.name))
                            
            steps += 1
            
            #####################################################
            # lr warmup
            # NOTE: the warmup phase run must not be terminated and be fully run during the first 4h run.
            # if the job is killed during the warmup phase, the peak learning rate will be set incorrectly (lower than the one specified)
            # I know it's ugly!
            #####################################################
            if WARMUP_PHASE:
                with warmup_g.dampening():
                    scheduler_g.step()
                with warmup_d.dampening():
                    scheduler_d.step()
            else:
                scheduler_g.step()
                scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
        
def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)

    # parser.add_argument('--input_wavs_dir', default='LibriTTS')
    # parser.add_argument('--input_training_file', default='LibriTTS/train-full.txt')
    # parser.add_argument('--input_validation_file', default='LibriTTS/val-full.txt')
    # parser.add_argument('--list_input_unseen_wavs_dir', nargs='+', default=['LibriTTS', 'LibriTTS'])
    # parser.add_argument('--list_input_unseen_validation_file', nargs='+', default=['LibriTTS/dev-clean.txt', 'LibriTTS/dev-other.txt'])

    parser.add_argument('--checkpoint_path', default='exp/bigvgan')

    parser.add_argument('--config', default='configs/bigvgan_22khz_80band.json')

    parser.add_argument('--dataset_config', default='datasets/libritts.json')
    
    # for fine-tuning, not used for now
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--input_mels_dir', default='ft_dataset')

    parser.add_argument('--training_epochs', default=100000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=50000, type=int)

    parser.add_argument('--freeze_step', default=0, type=int,
                        help='freeze D for the first specified steps. G only uses regression loss for these steps.')

    parser.add_argument('--debug', default=False, type=bool,
                        help="debug mode. skips validation loop throughout training and check data pass (does not train the model)")
    parser.add_argument('--evaluate', default=False, type=bool,
                        help="only run evaluation from checkpoint and exit")
    parser.add_argument('--eval_subsample', default=5, type=int,
                        help="subsampling during evaluation loop")
    parser.add_argument('--skip_seen', default=False, type=bool,
                        help="skip seen dataset. useful for test set inference")
    parser.add_argument('--save_audio', default=False, type=bool,
                        help="save audio of test set inference to disk")
    
    # for overriding hyperparams
    parser.add_argument('--params', nargs='+', default=[],
                        help="hyperparameter override for model config (NOT dataset_config)")

    a = parser.parse_args()

    with open(a.config) as f:
        config = f.read()
    with open(a.dataset_config) as f:
        dataset_config = f.read()

    json_config = json.loads(config)
    json_dataset_config = json.loads(dataset_config)
                
    # identify model_type: currently "vocoder" or "autoencoder".
    # for backward compatibility, this will add model_type="vocoder" to h if not found
    if not "model_type" in json_config.keys():
        json_config["model_type"] = "vocoder"
    assert json_config["model_type"] in ["vocoder", "autoencoder", "vae"], f"unknown model_type {json_config['model_type']}"
    
    # override json_config with hparams
    if a.params != []:
        print("##############################################")
        print("WARNING: Overriding hyperparameres from --params!#")
        json_config = update_params(json_config, a.params)
        print("##############################################")
    if "freeze_step" in json_config.keys(): # move --params freeze_step to a if set
        a.freeze_step = json_config["freeze_step"]
    if a.freeze_step != 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"WARNING: --freeze_step set to {a.freeze_step}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    h = AttrDict({**json_config, **json_dataset_config})
    h.debug = a.debug # add debug flag to h as well
    h.data["debug"] = a.debug
    # override randomize TCP port (before seed_everything)
    h.dist_config['dist_url'] = "tcp://localhost:" + str(random.randint(11111, 55555))

    build_env(json_config, 'config.json', a.checkpoint_path)
    build_env(json_dataset_config, 'dataset_config.json', a.checkpoint_path)
    
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    # also randomize seed, we cannot pass full epoch using 4 hour jobs
    seed = random.randint(0, 99999)
    print(f"INFO: applying seed_everything() using random seed {seed}")
    seed_everything(seed)
    
    if torch.cuda.is_available():
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
