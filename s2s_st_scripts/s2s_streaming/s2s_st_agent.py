"""
SimulEval agent for streaming full-duplex speech-to-speech / speech-to-text
translation using NeMo's DuplexS2SSpeechDecoderModel.

Author: Amir Hussein (NVIDIA / Johns Hopkins University)

Basic usage:
    simuleval --agent s2s_st_agent.py \
              --source-segment-size 80 \
              --source source.txt \
              --target target.txt \
              --output output \
              --quality-metrics BLEU \
              --config-path /path/to/exp_config.yaml \
              --model-path  /path/to/checkpoints/stepXXXX.ckpt
"""

import os
import re
import contextlib
from time import perf_counter
import yaml

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from omegaconf import OmegaConf
from nemo.core import typecheck
from nemo.collections.speechlm2 import DuplexS2SSpeechDecoderModel2
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str

import logging
logger = logging.getLogger(__name__)
import os

#simuleval --agent s2s_st_agent.py --source-segment-size 80 --source source.txt --target target.txt --output output --quality-metrics BLEU --config-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa4/exp_config.yaml --model-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa4/checkpoints/step11001.ckpt
#simuleval --agent s2s_st_agent.py --source-segment-size 80 --source source.txt --target target.txt --output output_target_aligned --quality-metrics BLEU --config-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa_target_aligned_4nodes_encoder_70_st_concat_v_mfa_target_aligned_fixed_prompt_dur/exp_config.yaml --model-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa_target_aligned_4nodes_encoder_70_st_concat_v_mfa_target_aligned_fixed_prompt_dur/checkpoints/step8001.ckpt
# simuleval --agent s2s_st_agent.py --source-segment-size 80 --source source.txt --target target.txt --output output_target_aligned --quality-metrics BLEU --config-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa_target_aligned_4nodes_encoder_70_st_concat_v_mfa_target_aligned_fixed_prompt_dur/exp_config.yaml --model-path /lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa_target_aligned_4nodes_encoder_70_st_concat_v_mfa_target_aligned/checkpoints/step17002.ckpt

# simuleval --agent s2s_st_agent.py --latency-metrics LAAL MAAL --source-segment-size 80 --source es-US_en-US/source.txt --target es-US_en-US/target.txt --output es-US_en-US/target_aligned1.5s --ctm-path es-US_en-US/words.combined.ctm  --t2t-align-path es-US_en-US/ --quality-metrics BLEU --config-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_target_aligned_4nodes_encoder_70_st_concat_v_mfa7_target_aligned_new_es-US/exp_config.yaml --model-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_target_aligned_4nodes_encoder_70_st_concat_v_mfa7_target_aligned_new_es-US/checkpoints/step14003.ckpt
#simuleval --agent s2s_st_agent.py --latency-metrics LAAL MAAL --source-segment-size 80 --source fr-FR_en-US/source.txt --target fr-FR_en-US/target.txt --output fr-FR_en-US/target_aligned1.5s --ctm-path fr-FR_en-US/words.combined.ctm  --t2t-align-path fr-FR_en-US/ --quality-metrics BLEU --config-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_fr_target_aligned/exp_config.yaml --model-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_fr_target_aligned/checkpoints/step10002.ckpt
#simuleval --agent s2s_st_agent.py --latency-metrics LAAL MAAL --source-segment-size 80 --source fr-FR_en-US/source.txt --target fr-FR_en-US/target.txt --output fr-FR_en-US/output_2s --ctm-path fr-FR_en-US/words.combined.ctm  --t2t-align-path fr-FR_en-US/ --quality-metrics BLEU --config-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_fr/exp_config.yaml --model-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_fr/checkpoints/step10002.ckpt
# simuleval --agent s2s_st_agent.py --latency-metrics LAAL MAAL --source-segment-size 80 --source de-DE_en-US/source.txt --target de-DE_en-US/target.txt --output de-DE_en-US/target_aligned1.5s --ctm-path de-DE_en-US/words.combined.ctm  --t2t-align-path de-DE_en-US/ --quality-metrics BLEU --config-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_de_target_aligned/exp_config.yaml --model-path /lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa6_de_target_aligned/checkpoints/step15003.ckpt

# simuleval --score-only --latency-metrics LAAL MAAL --source-segment-size 80 --source fr-FR_en-US/source.txt --target fr-FR_en-US/target.txt --output fr-FR_en-US/output_2s --ctm-path fr-FR_en-US/words.combined.ctm  --t2t-align-path fr-FR_en-US/ --quality-metrics BLEU 

os.environ["HF_HOME"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"

def synchronized_timer(description: str):
    @contextlib.contextmanager
    def timer_with_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = perf_counter() - start
        print(f"{description}: {elapsed_time:.4f} seconds")
    return timer_with_sync()

        
@dataclass
class S2TAgentStates(AgentStates):
    generated_text_tokens: torch.Tensor = None
    generated_audio_tokens: torch.Tensor = None
    cache: Optional[DynamicCache] = None
    spk_embedding: Optional[torch.Tensor] = None
    src_len: int = 0
    generated_text_list: list = field(default_factory=list)
    counter: int = 0
    first_call: bool = True
    audio_done:Optional[torch.Tensor] = None
    txt_done:Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        self.generated_text_tokens = None
        self.generated_audio_tokens = None
        self.cache = None
        self.spk_embedding = None
        self.audio_done = None
        self.txt_done = None
        self.src_len = 0
        self.counter = 0
        self.generated_text_list = []
        self.first_call = True

@entrypoint
class DuplexModel(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        # transformers.set_seed(42)

        self.source_segment_size = args.source_segment_size
        self.DTYPE = torch.float32
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CODEC_TOKEN_HISTORY_SIZE = 70
        self.INPUT_AUDIO_CHUNK_SIZE_SEC = 0.08
        self.USER_SAMPLE_RATE = 16000
        self.AGENT_SAMPLE_RATE = 22050
        self.input_audio_chunk_size_samples = int(self.INPUT_AUDIO_CHUNK_SIZE_SEC * self.USER_SAMPLE_RATE)
        self.AUDIO_BUFFER_SIZE = 128 * self.input_audio_chunk_size_samples
        self.prompt_dur = 0.5 * self.USER_SAMPLE_RATE
        
        # simuleval
        self.sent_idx = 0

        # # cache
        # self.llm_cache_truncated = 0  
        # self.llm_cache_kept_initial = args.llm_cache_kept_initial
        # self.llm_cache_kept_end = args.llm_cache_kept_end
        
        # model
        print("Initializing DuplexModel...")
        self.model_path = args.model_path
        self.config_path = args.config_path
        self.model = None
        self.tokenizer = None
        self.initialize()
        self.spk_embedding = None
        print("DuplexModel initialized successfully.")

        # others
        self.verbose = args.verbose



    def initialize_model_stuff(self):
        """
        Loads the model configuration, instantiates the model, and loads the
        pre-trained weights from a checkpoint, RESPECTING HETEROGENEOUS DTYPES.
        """
        print("Loading model configuration and weights...")
        with open(self.config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        if 'hyper_parameters' in cfg_dict:
            cfg_dict = cfg_dict['hyper_parameters']


        self.model = DuplexS2SSpeechDecoderModel2(cfg_dict)


        checkpoint = torch.load(self.model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}


        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.DEVICE)
        # self.model.to(DTYPE)
        self.model.eval()
        self.model.on_train_epoch_start()
         
        print("--- Verifying Dtypes of Loaded Modules ---")
        try:
            print(f"LLM (llm) dtype: {next(self.model.llm.parameters()).dtype}")
            print(
                f"Speech Generation (speech_generation) dtype: {next(self.model.speech_generation.parameters()).dtype}")
            print(f"Perception (perception) dtype: {next(self.model.perception.parameters()).dtype}")
        except StopIteration:
            print("A module has no parameters.")
        print("-----------------------------------------")

        self.tokenizer = self.model.tokenizer
        print(f"Model configured with mixed precision on {self.DEVICE}.")

    def initialize(self):
        """
        High-level initialization function.
        """
        self.initialize_model_stuff()
        self.clear_input_history()

    def clear_input_history(self, gt_text="", gt_num_unk_prepend=None):
        """
        Resets the state of the model for a new inference session. This is
        called at the beginning of each new interaction.
        """

        # --- State for Autoregressive Generation ---
        # KV-cache for the language model
        # self.cache = DynamicCache()
        # Reset the cache for the speech decoder module
        self.model.speech_generation.reset_input_and_kv_cache(use_cache=True)

        # Tensors to store the full sequence of generated tokens
        # self.generated_text_tokens = torch.empty(1, 0, device=DEVICE, dtype=torch.long)
        # # self.generated_audio_tokens = torch.empty(1, CODEC_TOKEN_HISTORY_SIZE, self.model._num_codebooks, device=DEVICE,
        # #                                           dtype=torch.long)
        # self.generated_audio_tokens = torch.empty(
        #                                 1, 0, self.model._num_codebooks,  # [B, T, K], T = 0 initially
        #                                 device=DEVICE,
        #                                 dtype=torch.long,)

        # Current length of the generated sequence
        self.context_length = 0
        self.llm_cache_truncated = 0        

        # gt_text override is not implemented for the new model but the
        # interface is kept for compatibility.
        if gt_text:
            print(f"Warning: gt_text override is not supported in this version.")

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Gets the embedding for the beginning-of-sequence token.
        In the new model, this is simulated by using the PAD token, as per model.py.
        """
        text_bos = torch.full((1,), fill_value=self.model.text_pad_id, device=self.DEVICE)
        input_embeds = self.model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.DTYPE)
    
    def update_prompt(self, prompt,sr):
        prompt_len = torch.tensor([prompt.shape[-1]], device=prompt.device, dtype=torch.int32)
        speaker_encoder_emb = self.model.speech_generation.get_speaker_embedding(
                        prompt, prompt_len, sr
                    )
                
        self.model.speech_generation.update_inference_speaker_embedding_from_embedding(
            speaker_encoder_emb
        )
        return speaker_encoder_emb
    
    def decode_audio_tokens(self, codes):
        """
        Converts a batch of audio codec tokens into a waveform.
        """
        if codes.numel() == 0:
            return torch.tensor([[]], device=self.DEVICE)

        # The new model has a helper function to replace control codes

        codes = replace_control_speech_codes(codes, self.model._control_codes)

        # The audio_codec expects float32 precision
        with torch.cuda.amp.autocast(enabled=False):
            codes = codes.to(torch.long)
            # The codec's `decode` method expects tokens in shape (B, K, T)
            codes_transposed = codes.transpose(1, 2)

            wav, wav_len = self.model.audio_codec.decode(tokens=codes_transposed, tokens_len=torch.tensor([codes.shape[1]], device=self.DEVICE))

        return wav, wav_len

    @staticmethod
    def add_args(parser):
        # load
        parser.add_argument("--config-path", type=str, default='conf')
        parser.add_argument("--model-path", type=str, default=None)

        # others
        parser.add_argument("--verbose", type=int, default=1)

    def build_states(self):
        return S2TAgentStates(
            generated_text_tokens = None,
            generated_audio_tokens = None,
            cache = None,
            src_len=0,
            spk_embedding = None,
            counter = 0,
            generated_text_list = [],
            audio_done = None,
            txt_done = None,
        )

    @typecheck.disable_checks()
    def _prepare_speech(self, states):        
        # Only tensorize the new part
        states.src_len = len(states.source)
        audio_len_tensor = torch.tensor([len(states.source)], device=self.DEVICE)
        audio_to_encode = torch.tensor(states.source, device=self.DEVICE, dtype=torch.float32).view(1, -1)
 
        if audio_to_encode.shape[1] <= self.prompt_dur:
        
            states.spk_embedding = self.update_prompt(audio_to_encode, self.USER_SAMPLE_RATE)
 
        # if audio_in_chunk.shape[1] < min_audio_samples_for_next_inference:
        #     # Not enough audio yet, return empty
        #     return "", np.array([[]], dtype=np.float32)

        # --- 1. Perception (Audio Encoding) ---
        
        source_encoded, _, asr_emb = self.model.perception(
            input_signal=audio_to_encode,
            input_signal_length=audio_len_tensor,
            return_encoder_emb=True,
        )
        source_encoded = source_encoded.to(self.DTYPE)
        asr_emb = asr_emb.to(self.DTYPE)
        return source_encoded

    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None:
            states = self.states

        if states.cache == None:
            states.cache =  DynamicCache() 
                  
        if states.generated_text_tokens == None:
            states.generated_text_tokens = torch.empty(1, 0, device=self.DEVICE, dtype=torch.long)
        if states.generated_audio_tokens == None:
            states.generated_audio_tokens = torch.empty(1, 0, self.model._num_codebooks, device=self.DEVICE, dtype=torch.long)
        if states.audio_done == None:
            states.audio_done = torch.zeros([1], dtype=torch.bool, device=self.DEVICE)
            states.txt_done = torch.zeros([1], dtype=torch.bool, device=self.DEVICE)

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate
        # if the source audio finished and length is very short don't try to decode
        if states.source_finished and length_in_seconds < 0.2:
            if self.verbose:
                print(f"[WRITE] sent={self.sent_idx} very-short-utt, src_len={len(states.source)},sr={states.source_sample_rate}, content='', finished=True")
            return WriteAction(content="", finished=True)
        
        # read a chunk or 80ms
        if not states.source_finished:
            if length_in_seconds*1000 < self.source_segment_size:
                if self.verbose:
                    print(f"[READ]  sent={self.sent_idx} src_len={len(states.source)} (len_sec={length_in_seconds:.3f} < {self.source_segment_size}ms)")
                return ReadAction()

        # ensure you're accurately profiling the real cost especially for GPU which are asynchronous by default
        with synchronized_timer('generate'):
            source_encoded = self._prepare_speech(states)
            if states.first_call:
                self.clear_input_history()
                states.first_call = False
                last_emb = self._get_bos_embedding()
                last_audio_tokens = torch.full(
                    (1, 1, self.model._num_codebooks),
                    fill_value=self.model.speech_delay_id,
                    device=self.DEVICE,
                    dtype=torch.long,
                )
            else:
                # Subsequent steps: use embeddings of previously generated tokens
                last_text_token = states.generated_text_tokens
                last_emb = self.model.embed_tokens(last_text_token).to(self.DTYPE)
                last_audio_tokens = states.generated_audio_tokens[:, -1:, :]

            current_input_embeds = (
                    source_encoded[:, -2].unsqueeze(1) * self.model.cfg.get("duplex_user_channel_weight", 1.0))

            current_input_embeds += last_emb.unsqueeze(1)  

            source_encoded_step = source_encoded[:, -2].unsqueeze(1)

            ans = self.model.forward(
                input_embeds=current_input_embeds, cache=states.cache, input_audio_tokens=last_audio_tokens,
                modality_adapter_emb=source_encoded_step, asr_emb=None, speaker_encoder_emb=states.spk_embedding, seq_mask=None
            )
            states.cache = ans["cache"]
            next_text_token = ans["text_logits"][:, -1].argmax(dim=-1)
            next_audio_tokens = ans["audio_logits"][:, -1].argmax(dim=-1)

            states.generated_text_tokens = next_text_token
            states.generated_audio_tokens = torch.cat((states.generated_audio_tokens, next_audio_tokens.unsqueeze(1)), axis=1)
            if states.generated_audio_tokens.size(1) > self.CODEC_TOKEN_HISTORY_SIZE:
                states.generated_audio_tokens = states.generated_audio_tokens[:, -self.CODEC_TOKEN_HISTORY_SIZE:, :]

            new_text_tokens = states.generated_text_tokens.unsqueeze(1).cpu()
            text_output_str = tokens_to_str(new_text_tokens, torch.tensor([1]), tokenizer=self.tokenizer, pad_id=self.model.text_pad_id)
            txt_translation = ''.join(text_output_str)
            states.generated_text_list.append(txt_translation)

            decoded_audio_wav, wav_len = self.decode_audio_tokens(states.generated_audio_tokens)
            new_audio_samples = self.model.audio_codec.samples_per_frame

            if decoded_audio_wav.shape[1] > 0:
                pad = decoded_audio_wav.shape[1] - wav_len.item()
                if pad != 0:
                    assert pad > 0, "Padding should be positive"
                    decoded_audio_wav = decoded_audio_wav[:, :-pad]
                audio_output_wav = decoded_audio_wav[:, -new_audio_samples:]
            else:
                audio_output_wav = np.array([[]], dtype=np.float32)
            
            speech_done = (next_audio_tokens == self.model.speech_eos_id).any(dim=1)
            text_done = (next_text_token.squeeze(0) == self.model.text_eos_id)
            newly_speech_done = (~states.audio_done) & (speech_done)
            newly_text_done = (~states.txt_done) & (text_done)
            states.audio_done |= newly_speech_done
            states.txt_done |= newly_text_done
        if self.verbose and not states.source_finished:
            print("".join(states.generated_text_list))
        if states.audio_done.all() and states.txt_done.all():
            states.source_finished = True
        if states.source_finished:
            self.sent_idx += 1
            if self.verbose:
                print(f"[INFO] sentence finished → sent_idx={self.sent_idx}")
        
        if txt_translation != '' or states.source_finished:
            if self.verbose:
                print(
                    f"[WRITE] sent={self.sent_idx} src_len={len(states.source)} "
                    f"sr={states.source_sample_rate}, text='{txt_translation}', "
                    f"finished={states.source_finished}"
                )
            return WriteAction(
                content=txt_translation,
                finished=states.source_finished,
            )
        else:
            if self.verbose:
                print(
                    f"[READ]  sent={self.sent_idx} src_len={len(states.source)} "
                    f"sr={states.source_sample_rate}, (no text yet, decoding step done)"
                )
            return ReadAction()