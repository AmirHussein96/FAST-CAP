# SCRIPT FOR QWEN CKPT

import torch
import yaml
from omegaconf import OmegaConf
import numpy as np
import librosa
import time
from transformers import DynamicCache
import re
import os
import sys
from lhotse import  CutSet
import argparse
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.audio.parts.utils.resampling import resample
import json
from tqdm import tqdm
# from utils import setup_deterministic_pytorch

# setup_deterministic_pytorch(seed=42, force_fp32_precision=False)



# sys.path.append('/models/ckpt_qwen_jul11/NeMo-rebased-main')
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE"


from nemo.collections.speechlm2 import DuplexS2SSpeechDecoderModel2
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str

# --- Configuration ---
# DTYPE = torch.bfloat16
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Inference Parameters ---
# These parameters are preserved from your original script.
# How many chunks of audio we group together for inference
N_CHUNKS_PER_INFERENCE = 1
# How many chunks of response audio we will generate. 1250 * 0.08s = 100 seconds
MAX_LEN = 1250
# The size of the audio history (in chunks) to feed to the audio codec for decoding
CODEC_TOKEN_HISTORY_SIZE = 60
# The size of each audio chunk in seconds
INPUT_AUDIO_CHUNK_SIZE_SEC = 0.08
# The sample rate of the user audio and agent
USER_SAMPLE_RATE = 16000
AGENT_SAMPLE_RATE = 22050
# The size of each input chunk in samples
input_audio_chunk_size_samples = int(INPUT_AUDIO_CHUNK_SIZE_SEC * USER_SAMPLE_RATE)

# prompt duration
prompt_length = 3*USER_SAMPLE_RATE



class DuplexModel:
    """
    Wrapper around the DuplexS2SSpeechDecoderModel, adapted for streaming inference.
    It maintains the structure of the original inference script's DuplexModel class.
    """

    def __init__(self, model_path: str, config_path: str):
        """
        Initializes the model and sets up the necessary components.
        Args:
            model_path (str): Path to the pre-trained .ckpt model file.
            config_path (str): Path to the model's .yaml configuration file.
        """
        print("Initializing DuplexModel...")
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.initialize()
        self.spk_embedding = None
        print("DuplexModel initialized successfully.")

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

        self.model.to(DEVICE)
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
        print(f"Model configured with mixed precision on {DEVICE}.")

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
        # Counter for tracking inference steps (chunks)
        self.counter = 0

        # Buffer to accumulate incoming audio chunks
        self.audio_in_buffer = torch.tensor([[]], dtype=torch.float32, device=DEVICE)
        self.audio_prompt_buffer = torch.tensor([[]], dtype=torch.float32, device=DEVICE)

        # --- State for Autoregressive Generation ---
        # KV-cache for the language model
        self.cache = DynamicCache()
        # Reset the cache for the speech decoder module
        self.model.speech_generation.reset_input_and_kv_cache(use_cache=True)

        # Tensors to store the full sequence of generated tokens
        self.generated_text_tokens = torch.empty(1, MAX_LEN, device=DEVICE, dtype=torch.long)
        self.generated_audio_tokens = torch.empty(1, MAX_LEN, self.model._num_codebooks, device=DEVICE,
                                                  dtype=torch.long)

        # Current length of the generated sequence
        self.context_length = 0

        self.speech_state = torch.zeros(1, device=DEVICE, dtype=torch.long)        

        # gt_text override is not implemented for the new model but the
        # interface is kept for compatibility.
        if gt_text:
            print(f"Warning: gt_text override is not supported in this version.")

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Gets the embedding for the beginning-of-sequence token.
        In the new model, this is simulated by using the PAD token, as per model.py.
        """
        text_bos = torch.full((1,), fill_value=self.model.text_pad_id, device=DEVICE)
        input_embeds = self.model.embed_tokens(text_bos)
        return input_embeds.to(dtype=DTYPE)


    def execute(self, audio_in_chunk, audio_done, txt_done, predefined_spk, reset):
        """
        Processes a chunk of audio and returns the generated text and audio.
        This is the main entry point for streaming inference.
        """
        # --- Input Validation and Reset ---
        if reset:
            if audio_in_chunk.size > 0:
                raise ValueError("Cannot request reset if audio_in_chunk is not empty")
            self.clear_input_history()
            print('Input history cleared.')
            return "", np.array([[]], dtype=np.float32), audio_done, txt_done

        if not audio_in_chunk.size and not reset:
            raise ValueError("Cannot have both empty audio_in_chunk and reset=False")

        # --- Audio Buffer Management ---
        audio_in_chunk = torch.tensor(audio_in_chunk, device=DEVICE, dtype=torch.float32).view(1, -1)
        self.audio_in_buffer = torch.cat((self.audio_in_buffer, audio_in_chunk), axis=1)

        if self.audio_in_buffer.shape[1] <= prompt_length and not predefined_spk:
        
            self.spk_embedding = self.update_prompt(self.audio_in_buffer, USER_SAMPLE_RATE)
        # if (not self.prompt_set) and (self.audio_in_buffer.shape[1] >= prompt_length):
        #     # Use only the first `prompt_length` samples for speaker embedding
        #     prompt = self.audio_in_buffer[:, :prompt_length]
        #     self.spk_embedding = self.update_prompt(prompt)
        #     self.prompt_set = True
        # Check if we have enough audio to process the next N_CHUNKS_PER_INFERENCE
        min_audio_samples_for_next_inference = (self.counter + N_CHUNKS_PER_INFERENCE) * input_audio_chunk_size_samples
        if self.audio_in_buffer.shape[1] < min_audio_samples_for_next_inference:
            # Not enough audio yet, return empty
            print("Not enough audio yet, return empty")
            return "", np.array([[]], dtype=np.float32), audio_done, txt_done

        # start_perception = time.time()

        # --- 1. Perception (Audio Encoding) ---
        # Get embeddings for the entire buffered audio up to the point we need.
        buffer_len_tensor = torch.tensor([min_audio_samples_for_next_inference], device=DEVICE)

        # The perception module expects float32
        audio_to_encode = self.audio_in_buffer[:, :min_audio_samples_for_next_inference].to(torch.float32)

        source_encoded, _, asr_emb = self.model.perception(
            input_signal=audio_to_encode,
            input_signal_length=buffer_len_tensor,
            return_encoder_emb=True,
        )
        source_encoded = source_encoded.to(DTYPE)
        asr_emb = asr_emb.to(DTYPE)
        # print(f"Time for perception: {time.time() - start_perception:.4f}s")

        # --- 2. Autoregressive Generation ---
        # We generate N_CHUNKS_PER_INFERENCE tokens in a loop.
        start_idx = self.context_length
        end_idx = self.context_length + N_CHUNKS_PER_INFERENCE

        force_speech_state_flag = self.model.cfg.get('inference_force_speech_state', False)

        for ar_step in range(start_idx, end_idx):
            start_ar_step = time.time()

            # Prepare inputs for the current step
            if ar_step == 0:
                # First step: use BOS embedding and speech_delay token
                last_emb = self._get_bos_embedding()
                last_audio_tokens = torch.full(
                    (1, 1, self.model._num_codebooks),
                    fill_value=self.model.speech_delay_id,
                    device=DEVICE,
                    dtype=torch.long,
                )
            else:
                # Subsequent steps: use embeddings of previously generated tokens
                last_text_token = self.generated_text_tokens[:, ar_step - 1]
                last_emb = self.model.embed_tokens(last_text_token).to(DTYPE)
                last_audio_tokens = self.generated_audio_tokens[:, ar_step - 1: ar_step, :]

            
             # Combine text embedding with the perception output for this timestep
   
                #print(source_encoded.shape, ar_step)
            current_input_embeds = (
                    source_encoded[:, ar_step: ar_step + 1] * self.model.cfg.get("duplex_user_channel_weight", 1.0))
                #print(current_input_embeds.shape) 1 x 1 x 2048
            # else:
            #     current_input_embeds = source_encoded[:, -(N_CHUNKS_PER_INFERENCE-(ar_step - self.context_length))-1] * self.model.cfg.get("duplex_user_channel_weight", 1.0)
            #     current_input_embeds = (current_input_embeds.unsqueeze(1))  # Ensure correct shape [B, 1, D]

            current_input_embeds += last_emb.unsqueeze(1)  # Ensure correct shape [B, T, D]


            source_encoded_step = source_encoded[:, ar_step: ar_step + 1]
            asr_emb_step = asr_emb[:, ar_step: ar_step + 1]

            ans = self.model.forward(
                input_embeds=current_input_embeds, cache=self.cache, input_audio_tokens=last_audio_tokens,
                modality_adapter_emb=source_encoded_step, asr_emb=None, speaker_encoder_emb=self.spk_embedding, seq_mask=None
            )
            self.cache = ans["cache"]
            next_text_token = ans["text_logits"][:, -1].argmax(dim=-1)
            next_audio_tokens = ans["audio_logits"][:, -1].argmax(dim=-1)
            if force_speech_state_flag:
                self.speech_state = torch.where(next_text_token == self.model.text_bos_id, 1, self.speech_state)
                self.speech_state = torch.where(next_text_token == self.model.text_eos_id, 0, self.speech_state)
                is_silent = (self.speech_state == 0)
                if ar_step > 0 and is_silent.item():
                    silence_audio_token = self.generated_audio_tokens[:, 0, :]
                    next_audio_tokens = torch.where(is_silent.unsqueeze(-1), silence_audio_token, next_audio_tokens)
            self.generated_text_tokens[:, ar_step] = next_text_token
            self.generated_audio_tokens[:, ar_step, :] = next_audio_tokens

            # print(f"Time for AR step {ar_step}: {time.time() - start_ar_step:.4f}s")

        # --- 3. Decode and Return Outputs ---
        # The model has now generated tokens from index `start_idx` to `end_idx - 1`

        # Decode the newly generated text tokens
        new_text_tokens = self.generated_text_tokens[:, start_idx:end_idx].cpu()
        text_output_str = tokens_to_str(new_text_tokens, torch.tensor([end_idx-start_idx]), tokenizer=self.tokenizer, pad_id=self.model.text_pad_id)

        # Decode the newly generated audio tokens
        # We provide a history of tokens to the codec for better quality
        history_start_idx = max(0, end_idx - CODEC_TOKEN_HISTORY_SIZE)
        audio_tokens_to_decode = self.generated_audio_tokens[:, history_start_idx:end_idx, :]

        # time_start_decode_audio = time.time()

        decoded_audio_wav, wav_len = self.decode_audio_tokens(audio_tokens_to_decode)
        
        # print(f'Time to decode audio tokens: {time.time() - time_start_decode_audio:.4f}s')
        # We decoded a segment with history, but we only want to return the new part
        new_audio_samples = self.model.audio_codec.samples_per_frame * N_CHUNKS_PER_INFERENCE


        # Ensure the decoded audio is not empty before slicing
        if decoded_audio_wav.shape[1] > 0:
            pad = decoded_audio_wav.shape[1] - wav_len.item()
            if pad != 0:
                assert pad > 0, "Padding should be positive"
                decoded_audio_wav = decoded_audio_wav[:, :-pad]
            audio_output_wav = decoded_audio_wav[:, -new_audio_samples:]
        else:
            audio_output_wav = np.array([[]], dtype=np.float32)
        # Update counters for the next execution
        self.counter += N_CHUNKS_PER_INFERENCE
        self.context_length += N_CHUNKS_PER_INFERENCE
        speech_done = (next_audio_tokens == self.model.speech_eos_id).any(dim=1)
        text_done = (next_text_token.squeeze(0) == self.model.text_eos_id)
        newly_speech_done = (~audio_done) & (speech_done)
        newly_text_done = (~txt_done) & (text_done)
        audio_done |= newly_speech_done
        txt_done |= newly_text_done
        # text_out = self.tokenizer.ids_to_tokens(new_text_tokens[0].tolist())

        return ''.join(text_output_str) , audio_output_wav.cpu().numpy(), audio_done, txt_done

    def decode_audio_tokens(self, codes):
        """
        Converts a batch of audio codec tokens into a waveform.
        """
        if codes.numel() == 0:
            return torch.tensor([[]], device=DEVICE)

        # The new model has a helper function to replace control codes

        codes = replace_control_speech_codes(codes, self.model._control_codes)

        # The audio_codec expects float32 precision
        with torch.cuda.amp.autocast(enabled=False):
            codes = codes.to(torch.long)
            # The codec's `decode` method expects tokens in shape (B, K, T)
            codes_transposed = codes.transpose(1, 2)

            wav, wav_len = self.model.audio_codec.decode(tokens=codes_transposed, tokens_len=torch.tensor([codes.shape[1]], device=DEVICE))

        return wav, wav_len

    def update_prompt(self, prompt,sr):
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.tensor(prompt, device=DEVICE, dtype=torch.float32).view(1, -1)
        prompt_len = torch.tensor([prompt.shape[-1]], device=prompt.device, dtype=torch.int32)
        speaker_encoder_emb = self.model.speech_generation.get_speaker_embedding(
                        prompt, prompt_len, sr
                    )
                
        self.model.speech_generation.update_inference_speaker_embedding_from_embedding(
            speaker_encoder_emb
        )
        return speaker_encoder_emb

# =========================================================================
#                   MAIN SCRIPT EXECUTION
# =========================================================================


def load_lhotse_shars(data_path):
    cuts = CutSet.from_shar(in_dir=data_path)
    return cuts



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=str, help="Path to SHAR dataset root")
    parser.add_argument("--output-dir", default="generated_eval", type=str)
    parser.add_argument("--predefined-spk", default=True, type=bool)
    return parser.parse_args()


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # MODEL_CHECKPOINT_PATH = "/lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa4/checkpoints/step11001.ckpt" # UPDATE THIS
    # MODEL_CONFIG_PATH = "/lustre/fsw/portfolios/llmservice/users/amhussein/s2s/exp/DFW_qwen_1b_st_concat_v_mfa2_4nodes_encoder_70_st_concat_v_mfa4/exp_config.yaml"  # UPDATE THIS
    MODEL_CHECKPOINT_PATH = "/lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_multiling_target_aligned2_8nodes_encoder_70_st_concat_v_mfa2_multiling_target_aligned2/checkpoints/step=24005.ckpt" # UPDATE THIS
    MODEL_CONFIG_PATH = "/lustre/fsw/portfolios/edgeai/users/amhussein/s2s_exp/DFW_qwen_1b_st_concat_v_mfa2_multiling_target_aligned2_8nodes_encoder_70_st_concat_v_mfa2_multiling_target_aligned2/exp_config.yaml"
    #INPUT_AUDIO_PATH = "/lustre/fsw/portfolios/llmservice/users/amhussein/untarred_data/covost_v2/es-US_en-US/0/common_voice_es_19141082.wav"  # UPDATE THIS
    # INPUT_AUDIO_PATH = "/lustre/fsw/portfolios/llmservice/users/amhussein/untarred_data/covost_v2/es-US_en-US/0/common_voice_es_19647509.wav"  # UPDATE THIS
    
    args = parse_args()
    cuts = load_lhotse_shars(args.data_path)
    

    print("loading model...")
    dm = DuplexModel(model_path=MODEL_CHECKPOINT_PATH, config_path=MODEL_CONFIG_PATH)
    asr_bleu = ASRBLEU(dm.model.cfg.scoring_asr).reset()
    print("ASR-BLEU initialized with ASR:", dm.model.cfg.scoring_asr)
    
    jsonl_f = open(f"{args.output_dir}.jsonl", "w")
    results = []
    # Load the input audio file
    # cuts = cuts.filter(lambda c: c.id == "common_voice_es_19865629")
    for i,cut in tqdm(enumerate(cuts), desc="Processing cuts"):
        print(f"processing cut: {cut.id}")
        c = cut.resample(USER_SAMPLE_RATE)
        y = c.recording.load_audio()
        target_text = " ".join(s.text for s in cut.supervisions if s.speaker in "agent")
        # print(f"Loaded audio file: {INPUT_AUDIO_PATH}")
        # prompt_length = min(prompt_duration * USER_SAMPLE_RATE, len(y))
        # prompt = y[:, 0:prompt_length]

        if args.predefined_spk:
            dm.spk_embedding = dm.update_prompt(y[:,:prompt_length], USER_SAMPLE_RATE)
        # The rest of this script is preserved from your original inference code.
        # It chunks the input audio and feeds it to the model.l

        # maximum length estimated from the input
        NUM_CHUNKS_TEST_AUDIO = int(y.shape[1]/(input_audio_chunk_size_samples))*1.5
        assert NUM_CHUNKS_TEST_AUDIO < MAX_LEN, f"NUM_CHUNKS_TEST_AUDIO must be less than MAX_LEN ({MAX_LEN})"

        # Variables for accumulating inference results
        generated_audio_list = []
        generated_text_list = []

        # Prepare audio signal (pad or trim)
        audio_signal = y.reshape(1, -1)
        padded_len_samples = int(NUM_CHUNKS_TEST_AUDIO * input_audio_chunk_size_samples)
        audio_signal_padded = np.zeros((1, padded_len_samples))

        current_len = audio_signal.shape[1]
        copy_len = min(padded_len_samples, current_len)
        audio_signal_padded[:, :copy_len] = audio_signal[:, :copy_len]

        # Pre-chunk the audio
        audio_signal_chunks = [
            audio_signal_padded[:, i:i + input_audio_chunk_size_samples]
            for i in range(0, audio_signal_padded.shape[1], input_audio_chunk_size_samples)
        ]

        # Reset the model state before starting
        dm.clear_input_history()

        print("\nStarting chunk-by-chunk inference loop...")
        total_processing_time = 0
        audio_done = torch.zeros([1], dtype=torch.bool, device=DEVICE)
        txt_done = torch.zeros([1], dtype=torch.bool, device=DEVICE)
        with torch.inference_mode():
            for i, audio_in_chunk in enumerate(audio_signal_chunks):

                chunk_start_time = time.time()
                # print(f"--- Processing chunk {i + 1}/{len(audio_signal_chunks)} ---")

                # Execute the model on the current chunk

                text_out, audio_out, audio_done, txt_done = dm.execute(audio_in_chunk=audio_in_chunk, audio_done=audio_done, txt_done=txt_done, predefined_spk=args.predefined_spk, reset=False, )
        
                if audio_done.all() and txt_done.all():
                    print("✅ EOS detected, stopping early.")
                    break
                chunk_end_time = time.time()
                processing_time = chunk_end_time - chunk_start_time
                total_processing_time += processing_time
                # print(f"Chunk {i + 1} processed in {processing_time:.4f}s")

                # Accumulate results if they are not empty
                if len(text_out) > 0:
                    generated_text_list.append(text_out)

                if audio_out.size > 0:
                    generated_audio_list.append(audio_out)


        print("\n--- Inference Complete ---")
        # Combine results

        # import pdb ;    pdb.set_trace()
        # text_token_seq = torch.cat(generated_text_list).view(-1).unsqueeze(0)
        # final_text = tokens_to_str(text_token_seq, torch.tensor([text_token_seq.shape[1]]), tokenizer=dm.tokenizer, pad_id=dm.model.text_pad_id)
        if generated_audio_list:
            tts_final_audio = np.concatenate(generated_audio_list, axis=1)
        else:
            tts_final_audio = np.array([[]])

        print(f"\nTotal Processing Time: {total_processing_time:.2f} seconds")
        print(f"\nFinal Generated Audio Shape: {tts_final_audio.shape}")

 
        # Convert collected token characters into final string
        final_text = "".join(generated_text_list)

        # Save to text file
        os.makedirs(args.output_dir, exist_ok=True)
        # text_output_path = "generated/generated_output.txt"
        # with open(text_output_path, "w") as f:
        #     f.write(final_text)

        # print(f"\nGenerated text saved to: {text_output_path}")
        print(f"Generated LLM text:\n{final_text}")
        asr_hyps = ""
        if tts_final_audio.size > 0:
            import soundfile as sf
            from numpy import pad
            output_audio_path = f"{args.output_dir}/quick_dev_{cut.id}.wav"
            c = cut.resample(AGENT_SAMPLE_RATE)
            user_for_save = c.recording.load_audio().astype(np.float32)
                # Scale user audio (make 2x quieter)
            user_scaled = user_for_save[0] * 0.5
            agent_audio = torch.from_numpy(tts_final_audio).to(device=DEVICE, dtype=torch.float32)
            agent_audio_len = torch.tensor([agent_audio.shape[1]], dtype=torch.long).to(DEVICE)

            asr_hyps = asr_bleu.update(
                    name=cut.id,
                    refs=[target_text],
                    pred_audio=resample(agent_audio, AGENT_SAMPLE_RATE, USER_SAMPLE_RATE),
                    pred_audio_lens=(agent_audio_len / AGENT_SAMPLE_RATE * USER_SAMPLE_RATE).to(torch.long),
                )
            tts_final_audio = tts_final_audio.reshape(-1)
            max_len = max(len(user_scaled), len(tts_final_audio))
            user_padded = np.zeros(max_len, dtype=np.float32)
            tts_final_audio_padded = np.zeros(max_len, dtype=np.float32)
            user_padded[:len(user_scaled)] = user_scaled
            tts_final_audio_padded[:len(tts_final_audio)] = tts_final_audio
            # if len(user_scaled) < max_len:
            #     user_scaled = pad(user_scaled, (0, max_len - len(user_scaled)))
            
            stereo_out = np.stack((user_padded, tts_final_audio_padded), axis=-1)
            sf.write(output_audio_path, stereo_out, AGENT_SAMPLE_RATE)
            # print(f"\nGenerated audio saved to: {output_audio_path}")

        # print(f"ASR TTS text:\n{asr_hyps[0]}")
        entry = {
            "target_text": target_text,
            "pred_text": final_text,
            "speech_pred_transcribed": asr_hyps[0],
            "audio_path": f"{cut.id}.wav",
        }
        results.append(entry)
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = f"{args.output_dir}/eval.jsonl"
    with open(jsonl_path, "w") as f:
        for obj in results:  # list of dictionaries
            f.write(json.dumps(obj, indent=4, ensure_ascii=False))

    print(f"Saved {len(results)} entries to {jsonl_path}")