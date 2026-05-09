import argparse
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
import os
from collections import defaultdict
from lhotse.utils import fastcopy
import numpy as np
import logging
from lhotse.supervision import AlignmentItem
import math
import io
import soundfile as sf
from lhotse import Recording, SupervisionSegment, CutSet, MonoCut
import numpy as np
import re

def slice_audio(cut, start_s, dur_s, recording_field=None):
    # Non-destructive: returns a new cut and only loads that region
    if recording_field is not None:
        target_audio = cut.custom[recording_field].load_audio()
        chunk = target_audio[:, int(start_s*cut.custom[recording_field].sampling_rate): int((start_s+dur_s)*cut.custom[recording_field].sampling_rate)]
    else:
        chunked_cut = cut.truncate(offset=start_s, duration=dur_s)
        chunk = chunked_cut.load_audio()
    return chunk

def parse_args():
    parser = argparse.ArgumentParser(description="Load data from tar and manifest using lhotse.")
    parser.add_argument('--src_tar', type=str, required=True, help='Path to the tar file containing the data.')
    parser.add_argument('--src_manifest', type=str, required=True, help='Path to the manifest file.')
    parser.add_argument('--src_ctm', type=str, required=True, help='Path to the ctm file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--txt_alignment_path', type=str, required=True, help='Path to alignment file (0-0 1-1 format).')
    parser.add_argument('--tgt_tar', type=str, required=True, help='Path to the tar file containing the target data.')
    parser.add_argument('--tgt_manifest', type=str, required=True, help='Path to the manifest file containing the target data.')
    parser.add_argument('--tgt_ctm', type=str, required=True, help='Path to the ctm file containing the target data.')
    parser.add_argument('--min_duration', type=float, default=1.0, help='Minimum duration in seconds.')
    parser.add_argument('--max_duration', type=float, default=30.0, help='Maximum duration in seconds.')
    parser.add_argument('--method', type=str, default='alignments', help='Method to combine the cuts.')
    parser.add_argument('--chunk_min_duration', type=float, default=0.56, help='min duration for combining source chunks.')
    return parser.parse_args()



def clean_txt(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_text_alignments(alignment_path: str, manifest_path: str):
    """
    This function loads the alignment file and returns a dictionary of alignment items with matching
     ids from the manifest.
    """
    manifest_name =os.path.splitext(os.path.basename(manifest_path))[0]
    alignment_file = os.path.join(alignment_path, f"{manifest_name}.awesome-align.out")
    alignment_ids_file = os.path.join(alignment_path, f"{manifest_name}.ids")
    alignment_dict = defaultdict(list)
    with open(alignment_file, 'r') as f, open(alignment_ids_file, 'r') as ids_f:
        for line in f:
            pairs = []
            for token in line.strip().split():
                try:
                    src, tgt = map(int, token.split('-'))
                    pairs.append((src, tgt))
                except ValueError:
                    continue
            alignment_dict[ids_f.readline().strip().split(".")[0]] = pairs
    return alignment_dict

def load_ctm_to_dict(ctm_path: str, cuts: CutSet) -> dict:
    alignment_dict = defaultdict(list)
    skipped = []
    with open(ctm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id, _, start, duration, symbol = parts[:5]
            if utt_id not in cuts:
                print(f"Utterance {utt_id} not found in cuts")
                if utt_id not in skipped:
                    skipped.append(utt_id)
                continue

            alignment_dict[utt_id].append(
                AlignmentItem(symbol=symbol, start=float(start), duration=float(duration))
            )
            if alignment_dict[utt_id][-1].end > cuts[utt_id].duration:
                # fix the last alignment item to be the duration of the cut
                alignment_dict[utt_id][-1] = AlignmentItem(symbol=alignment_dict[utt_id][-1].symbol, start=alignment_dict[utt_id][-1].start, duration=cuts[utt_id].duration - alignment_dict[utt_id][-1].start)
    print(f"Skipped {len(skipped)} utterances")
    return alignment_dict

def add_alignment_fn(alignment_dict):
    def add_alignment(sup):
        if sup.id in alignment_dict:
            return sup.with_alignment("word", alignment_dict[sup.id])
        return sup
    return add_alignment

def remove_extension_from_segment_id(segment):
    return fastcopy(segment, id=os.path.splitext(segment.id)[0])

def load_nemo_tarred_from_dir(manifest_path: str, tar_paths: str) -> CutSet:

    # Initialize iterator
    iterator = LazyNeMoTarredIterator(
                        manifest_path=manifest_path,
                        tar_paths=tar_paths,
                        allow_skipme=False,
                        shuffle_shards=False,
                    )
    return CutSet.from_cuts(iterator)

def add_src_traj(cut, tgt2src, frame_duration=0.08):
    
    sup = cut.supervisions[0]

    alignment = sorted(sup.alignment["word"], key=lambda x: x.start)
    offset = sup.start
    duration = sup.duration

    # 2. Pivots = unique sorted source positions from tgt2src
    pivot_src = sorted(set(tgt2src.values())) 
    # Compute chunking info
    src_traj = [[] for _ in range(len(pivot_src))]
    merged_alignments = [None for _ in range(len(pivot_src))]
  
    src_prev_step_idx = []
    traj_idx = 0
    for i,item in enumerate(alignment):
        # Calculate the frame index for the END of the word
        if i > pivot_src[traj_idx]:
            traj_idx += 1
        
        if merged_alignments[traj_idx] is None:
            merged_alignments[traj_idx] = item
            prev = merged_alignments[traj_idx]
            src_prev_step_idx.append(traj_idx)
        else:
            new_item = AlignmentItem(
                        symbol=" ".join([prev.symbol, item.symbol]),
                        start=prev.start,
                        duration=item.end - prev.start,
                    )
            merged_alignments[traj_idx] = new_item
            prev = merged_alignments[traj_idx]
            src_prev_step_idx.append(traj_idx)

        src_traj[traj_idx].append(item.symbol)
    # Convert to list of space-joined word strings per chunk
    src_traj = [' '.join(words) for words in src_traj]

    # Store in cut.custom
    # return fastcopy(cut, supervisions=[fastcopy(cut.supervisions[0], **cut.supervisions[0].alignment, "merged_alignments": merged_alignments)], custom={**cut.custom, "src_traj": src_traj, "src_alig_frames": tgt_alig_frames})
    return src_traj, merged_alignments


def mono_text_alignment(text_alignment, src_words_alignment, tgt_words_alignment):
    # text_alignment: list[(src_idx, tgt_idx)] 0-based
    # 1) sort by target then source
    
    if (len(src_words_alignment)-1, len(tgt_words_alignment) - 1) not in text_alignment:
        text_alignment.append((len(src_words_alignment) - 1, len(tgt_words_alignment) - 1))
    # remove alignments larger than number of words in source and target
    text_alignment = [(i, j) for (i, j) in text_alignment if i < len(src_words_alignment) and j < len(tgt_words_alignment)] 
    ali = sorted(text_alignment, key=lambda x: (x[1], x[0]))
    # 2) remove consecutive duplicates in tgt (keep last)
    ali_no_dup = []
    for a in ali:
        if ali_no_dup and ali_no_dup[-1][1] == a[1]:
            ali_no_dup[-1] = a
        else:
            ali_no_dup.append(a)

    # 3) enforce monotonic src
    for i, a in enumerate(ali_no_dup):
        if i == 0:
            continue
        ali_no_dup[i] = (max(a[0], ali_no_dup[i - 1][0]), a[1])

    tgt2src = {}
    for a in ali_no_dup:
        tgt2src[a[1]] = a[0]
    for i in range(len(tgt_words_alignment) - 1, -1, -1):
        if i not in tgt2src:
            tgt2src[i] = tgt2src[i + 1]

    return ali_no_dup, tgt2src


def recombine(src_merged_alignments, tgt_merged_alignments, src_traj, tgt_traj, chunk_min_duration=0.56):
    """
    Merge short source segments with their neighbors.
    All four lists are merged/deleted together to preserve alignment.
    """
    assert len(src_merged_alignments) == len(tgt_merged_alignments) == len(src_traj) == len(tgt_traj)
    i = 0
    while i < len(src_merged_alignments) and len(src_merged_alignments) > 1:
        src_item = src_merged_alignments[i]
        if src_item.duration >= chunk_min_duration:
            i += 1
            continue
        if i == len(src_merged_alignments) - 1:
            # last item, merge with previous
            j = i - 1
        else:
            j = i + 1
        keep = min(i, j)
        drop = max(i, j)
        # merge the two segments
        a_src = src_merged_alignments[keep]
        b_src = src_merged_alignments[drop]
        a_tgt = tgt_merged_alignments[keep]
        b_tgt = tgt_merged_alignments[drop]
     
        src_start = min(a_src.start, b_src.start)
        src_end = max(a_src.start + a_src.duration, b_src.start + b_src.duration)
        tgt_start = min(a_tgt.start, b_tgt.start)
        tgt_end = max(a_tgt.start + a_tgt.duration, b_tgt.start + b_tgt.duration)

      
        new_src_symbol = [a_src.symbol, b_src.symbol]
        new_tgt_symbol = [a_tgt.symbol, b_tgt.symbol]
        src_traj[keep] = src_traj[keep] + ' ' + src_traj[drop]
        tgt_traj[keep] = tgt_traj[keep] + ' ' + tgt_traj[drop]

        new_src_item = AlignmentItem(
                                symbol=" ".join(new_src_symbol),
                                start=src_start,
                                duration=src_end - src_start,)
        new_tgt_item = AlignmentItem(
                                symbol=" ".join(new_tgt_symbol),
                                start=tgt_start,
                                duration=tgt_end - tgt_start,)
    
        src_merged_alignments[keep] = new_src_item
        tgt_merged_alignments[keep] = new_tgt_item

        del src_merged_alignments[drop]
        del tgt_merged_alignments[drop]
        del src_traj[drop]
        del tgt_traj[drop]

    return src_merged_alignments, tgt_merged_alignments, src_traj, tgt_traj


def add_tgt_traj(cut, alignment_dict, tgt_cuts, frame_duration=0.08, chunk_min_duration=0.56):
    # target_duration_in_frames = cut.custom['target_audio'].duration / 0.08  # 1 frame = 80 ms

    text_alignment = alignment_dict[cut.id]
    tgt_cut = tgt_cuts[cut.id]
    tgt_words_alignment = sorted(tgt_cut.supervisions[0].alignment['word'], key=lambda x: x.start)
    src_words_alignment = sorted(cut.supervisions[0].alignment['word'], key=lambda x: x.start)
    # 1) get monotonic alignments

    ali_no_dup, tgt2src = mono_text_alignment(text_alignment, src_words_alignment, tgt_words_alignment)

    # 2) get source trajectory
    src_traj, src_merged_alignments = add_src_traj(cut, tgt2src, frame_duration=0.08)

    # 3) get the target trajectory
    idx2step = []
    for i in range(len(src_traj)):
        n_word = len(src_traj[i].split(' ')) - (src_traj[i] == '')
        idx2step.extend([i] * n_word)
    tgt_traj = [[] for _ in range(len(src_traj))]
    ##TODO: dont forget to clean the text

    tgt_merged_alignments = [None for _ in range(len(src_traj))]
    for i in range(len(tgt_words_alignment)):
        
        src_word_idx = tgt2src[i]
        src_step_idx = idx2step[src_word_idx]
        tgt_traj[src_step_idx].append(tgt_words_alignment[i].symbol)
        if tgt_merged_alignments[src_step_idx] is None:
            tgt_merged_alignments[src_step_idx] = tgt_words_alignment[i]
            prev_item = tgt_merged_alignments[src_step_idx]
        else:
            # merge words corresponding to the same input step
            new_item = AlignmentItem(
                        symbol=" ".join([prev_item.symbol, tgt_words_alignment[i].symbol]),
                        start=prev_item.start,
                        duration=tgt_words_alignment[i].end - prev_item.start,
                    )
            tgt_merged_alignments[src_step_idx] = new_item
            prev_item = tgt_merged_alignments[src_step_idx]

    tgt_traj = [
            ' '.join(t) for t in tgt_traj
        ]
    # 4) recombine short source segments
    src_merged_alignments, tgt_merged_alignments, src_traj, tgt_traj = recombine(src_merged_alignments, tgt_merged_alignments, src_traj, tgt_traj, chunk_min_duration=chunk_min_duration)
    cut.supervisions[0].alignment["word_target"] = tgt_cut.supervisions[0].alignment["word"]
    # return fastcopy(cut, supervisions=[fastcopy(cut.supervisions[0], alignment={**cut.supervisions[0].alignment, "word_target": tgt_cut.supervisions[0].alignment})],custom={**cut.custom, "tgt_traj": tgt_traj, "target_audio": tgt_cut.recording, "txt_org_ali": text_alignment,
    #  "txt_mono_ali": alignments_no_dup, "tgt_word_alignment": tgt_cut.supervisions[0].alignment, "tgt_alig_frames": tgt_alig_frames, "tgt_merged_alignments": tgt_merged_alignments})
    cut.target_audio = tgt_cut.recording
    return fastcopy(cut, custom={**cut.custom,
                                "src_traj": src_traj, "src_merged_alignments": src_merged_alignments,
                                "tgt_traj": tgt_traj, "txt_org_ali": text_alignment,
                                "txt_mono_ali": ali_no_dup, "target_recording_unaligned": True,
                                "tgt_duration": tgt_cut.duration,
                                "tgt_translation": tgt_cut.supervisions[0].text, 
                                "tgt_merged_alignments": tgt_merged_alignments})



def recording_from_numpy(waveform: np.ndarray, sr: int, rec_id: str = "rec_in_memory"):
    """
    Create a Lhotse Recording from a NumPy waveform entirely in memory.
    waveform shape: (1, num_samples) or (num_samples,) → mono.
    """
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # (1, num_samples)

    # Encode the waveform into WAV bytes in memory
    buffer = io.BytesIO()
    sf.write(buffer, waveform.T, sr, format="WAV")
    wav_bytes = buffer.getvalue()

    # Use Lhotse's from_bytes()
    recording = Recording.from_bytes(data=wav_bytes, recording_id=rec_id)
    return recording

def add_supervision(supervisions_list, id_, start, duration, text, speaker):
    supervisions_list.append(
        SupervisionSegment(
            id=id_,
            recording_id=id_,
            start=start,
            duration=duration,
            text=text,
            channel=0,
            speaker=speaker,
        )
    )

    
    return supervisions_list

def pad_right(arr: np.ndarray, T: int, value: float = 0.0) -> np.ndarray:

    assert arr.ndim == 2, "Array must be 2D"
    C, N = arr.shape
    if N >= T:
        return arr
    pad = np.full((C, (T - N)), value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)

def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)

def apply_gap_smoothing(gap, left, right):
    edge_len = int(min(len(left[0])*0.1, len(right[0])*0.1, len(gap[0])))
    window = np.hamming(2 * edge_len)
    left_edge = left[:, -edge_len:]
    right_edge = right[:, :edge_len]
    left_edge = left_edge[:, ::-1]
    right_edge = right_edge[:, ::-1]
    # optional rms normalization
    E_left = rms(left_edge)
    E_right = rms(right_edge)
    E_target = min(E_left, E_right)
    left_edge *= (E_target / (E_left + 1e-8))
    right_edge *= (E_target / (E_right + 1e-8))
    gap[:, -edge_len:] = right_edge * window[:edge_len]
    gap[:, :edge_len] = left_edge * window[edge_len:]  
    return gap

def smooth_gaps(tgt_rec_arr, tgt_rec_tags):
    new_arr = [tgt_rec_arr[0]]
    for i, (arr, tag) in enumerate(zip(tgt_rec_arr, tgt_rec_tags)):
        if i == 0:
            continue
        if tag == "gap" and len(arr[0]) > 0:
            smooth_gap = apply_gap_smoothing(arr, tgt_rec_arr[i-1], tgt_rec_arr[i+1])
            new_arr.append(smooth_gap)
        else:
            new_arr.append(arr)
    return new_arr

def combine_chunked_cuts(cuts):
    new_cuts = []
    tol = 0.01
    tgt_sampling_rate = cuts[0].custom['target_audio'].sampling_rate
    src_sampling_rate = cuts[0].sampling_rate
    
    for cut in cuts:
        tgt_full_dur = cut.custom["target_audio"].duration
        if cut.sampling_rate != tgt_sampling_rate:
            cut = cut.resample(tgt_sampling_rate)

        src_alignments = cut.custom['src_merged_alignments']
        tgt_alignments = cut.custom['tgt_merged_alignments']
        n_chunks = len(tgt_alignments)
        src_rec_arr = []
        tgt_rec_arr = []
        tgt_rec_tags = []
        assert len(src_alignments) == len(tgt_alignments), f"src_alignments and tgt_alignments have different lengths: {len(src_alignments)} != {len(tgt_alignments)}"
        supervisions = []
        tgt_offset = 0
        prev_tgt_end = 0
        for i, (src_alignment, tgt_alignment) in enumerate(zip(src_alignments, tgt_alignments)):
            # if last chunk take the last part of the target audio
            is_last = (i == n_chunks - 1)
            if tgt_alignment:
                if is_last:
                    tgt_end_time = tgt_full_dur
                else:
                    tgt_end_time = tgt_alignment.end
            if src_alignment and tgt_alignment:
                tgt_alig = [tgt_alignment.start, tgt_end_time]
                src_alig = [src_alignment.start, src_alignment.end]
                gap = src_alig[1] - tgt_offset
                L = int(round(gap * tgt_sampling_rate))
                # check if gap is larger that 1764 samples (i.e., 80ms at 22.05kHz)
                if L >= 1764:
                    tgt_rec_arr.append(np.zeros((1, L), dtype=np.float32))
                    tgt_rec_tags.append("gap")
                    tgt_offset = tgt_offset + gap
                tgt_start = tgt_offset
                tgt_end = tgt_start + (tgt_alig[1] - prev_tgt_end) 
                src_txt = src_alignment.symbol
                tgt_txt = tgt_alignment.symbol
                src_dur = src_alig[1]-src_alig[0]
                tgt_dur = tgt_alig[1]-prev_tgt_end
                tgt_slice = slice_audio(cut, prev_tgt_end, tgt_dur, recording_field="target_audio")

                # src_rec_arr.append(src_slice)
                tgt_rec_arr.append(tgt_slice)
                tgt_rec_tags.append("speech")
                supervisions = add_supervision(supervisions, cut.id, src_alig[0], src_dur, src_txt, speaker="user")
                supervisions = add_supervision(supervisions, cut.id, tgt_start, tgt_dur, tgt_txt, speaker="agent")
                tgt_offset = tgt_end 
                prev_tgt_end = tgt_alig[1]
                # add source and target supervisions to the cut
            elif src_alignment:
                src_alig = [src_alignment.start, src_alignment.end]
                src_dur = src_alig[1]-src_alig[0]
                src_txt = src_alignment.symbol
                supervisions = add_supervision(supervisions, cut.id, src_alig[0], src_dur, src_txt, speaker="user")
                # add source supervision to the cut
            elif tgt_alignment:
                tgt_alig = [tgt_alignment.start, tgt_end_time]
                tgt_dur = tgt_alig[1]-prev_tgt_end
                tgt_slice = slice_audio(cut, prev_tgt_end, tgt_dur, recording_field="target_audio")
                tgt_rec_arr.append(tgt_slice)
                tgt_rec_tags.append("speech")
                tgt_txt = tgt_alignment.symbol
                supervisions = add_supervision(supervisions, cut.id, tgt_offset, tgt_dur, tgt_txt, speaker="agent")
                offset = offset + tgt_dur
                prev_tgt_end = tgt_alig[1]
        src_rec_arr = cut.resample(tgt_sampling_rate).load_audio() # added resample
        src_dur = cut.duration
        if tgt_rec_tags.count("gap") > 1:
            tgt_rec_arr = smooth_gaps(tgt_rec_arr, tgt_rec_tags)
        tgt_rec_arr = np.concatenate(tgt_rec_arr, axis=1)
        if abs(len(src_rec_arr[0]) - len(tgt_rec_arr[0])) > 0:
            T = max(len(src_rec_arr[0]) , len(tgt_rec_arr[0]))
            src_rec_arr = pad_right(src_rec_arr, T)
            tgt_rec_arr = pad_right(tgt_rec_arr, T)
        total_dur = round(len(tgt_rec_arr[0]) / tgt_sampling_rate, 2)
        src_recording = recording_from_numpy(src_rec_arr, tgt_sampling_rate, rec_id=f"{cut.id}")
        tgt_recording = recording_from_numpy(tgt_rec_arr, tgt_sampling_rate, rec_id=f"{cut.id}")
        new_cut = MonoCut(id=cut.id, start=0, duration=total_dur, channel=0, recording=src_recording, supervisions=supervisions, custom={'src_duration': src_dur})
        new_cut = new_cut.resample(src_sampling_rate)
        new_cut.target_audio = tgt_recording
        new_cuts.append(new_cut)
    return CutSet.from_cuts(new_cuts)


def combine_cuts(src_cuts, tgt_cuts):
    cuts = []
    
    for src_cut in src_cuts:
        id_ = src_cut.id
        tgt_cut = tgt_cuts[id_]
        supervisions = []
        if src_cut.sampling_rate != tgt_cut.sampling_rate:
            src_cut = src_cut.resample(tgt_cut.sampling_rate)
        total_duration = src_cut.duration + tgt_cut.duration
        src_start, src_dur = src_cut.supervisions[0].start, src_cut.supervisions[0].duration
        tgt_start, tgt_dur = tgt_cut.supervisions[0].start, tgt_cut.supervisions[0].duration
        src_cut = src_cut.pad(duration=total_duration, direction="right")
        tgt_cut = tgt_cut.pad(duration=total_duration, direction="left")
        src_wav = src_cut.load_audio()
        tgt_wav = tgt_cut.load_audio()
        src_recording = recording_from_numpy(src_wav, src_cut.sampling_rate, rec_id=f"{id_}")
        tgt_recording = recording_from_numpy(tgt_wav, tgt_cut.sampling_rate, rec_id=f"{id_}")
        
        supervisions.append(
            SupervisionSegment(
                id=f"{id_}_src",
                recording_id=f"{id_}",
                start=src_start,
                duration=src_dur,
                channel=0,
                text=src_cut.supervisions[0].text,
                speaker="user",
            )
        )
        supervisions.append(
            SupervisionSegment(
                id=f"{id_}_tgt",
                recording_id=f"{id_}",
                start=src_start+ src_dur + tgt_start,
                duration=tgt_dur,
                channel=0,
                text=tgt_cut.supervisions[0].text,
                speaker="agent",
            )
        )

        new_cut = MonoCut(id=id_, start=src_start, duration=total_duration, channel=0, recording=src_recording, supervisions=supervisions, custom={'src_duration': src_dur})
        new_cut.target_audio = tgt_recording
        cuts.append(new_cut)
    return CutSet.from_cuts(cuts)

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    # Placeholder for loading data using lhotse
    logging.info(f"Tar file: {args.src_tar}")
    logging.info(f"Manifest file: {args.src_manifest}")
    logging.info(f"Ctm file: {args.src_ctm}")
    logging.info(f"Text alignment path: {args.txt_alignment_path}")
    logging.info(f"Target tar file: {args.tgt_tar}")
    logging.info(f"Target manifest file: {args.tgt_manifest}")
    logging.info(f"Target ctm file: {args.tgt_ctm}")


    src_cuts = load_nemo_tarred_from_dir(args.src_manifest, args.src_tar)
    shard_id = src_cuts[0].custom['shard_id']
    tgt_cuts = load_nemo_tarred_from_dir(args.tgt_manifest, args.tgt_tar)

    # remove the extension from the cut ids
    src_cuts = src_cuts.modify_ids(lambda id: os.path.splitext(id)[0])
    tgt_cuts = tgt_cuts.modify_ids(lambda id: os.path.splitext(id)[0])
    src_cuts = src_cuts.filter(lambda x: x.custom.get('_skipme') != 1)

    # remove the extension from the supervision ids
    src_cuts = src_cuts.map_supervisions(remove_extension_from_segment_id)
    tgt_cuts = tgt_cuts.map_supervisions(remove_extension_from_segment_id)

    # filter out the short and long cuts
    src_cuts = src_cuts.filter(lambda x: args.min_duration <= x.duration <= args.max_duration)
    tgt_cuts = tgt_cuts.filter(lambda x: 1 <= x.duration <= args.max_duration)

    if args.method == 'alignments':
        # loading text alignment
        alignment_text_dict = load_text_alignments(args.txt_alignment_path, args.src_manifest)


        src_alignment_dict = load_ctm_to_dict(args.src_ctm, src_cuts)
        tgt_alignment_dict = load_ctm_to_dict(args.tgt_ctm, tgt_cuts)
   
        # add the speech alignment to the cuts
        src_cuts = src_cuts.map_supervisions(add_alignment_fn(src_alignment_dict))
        tgt_cuts = tgt_cuts.map_supervisions(add_alignment_fn(tgt_alignment_dict))
        src_cuts = src_cuts.filter(lambda x: x.supervisions[0].alignment is not None)
        tgt_cuts = tgt_cuts.filter(lambda x: x.supervisions[0].alignment is not None)
    # trim the cuts to the alignments
    src_cuts = src_cuts.trim_to_supervisions(keep_overlapping=False)
    tgt_cuts = tgt_cuts.trim_to_supervisions(keep_overlapping=False)
    
    common_ids = set(src_cuts.ids) & set(tgt_cuts.ids)
    src_cuts = src_cuts.filter(lambda cut: cut.id in common_ids)
    tgt_cuts = tgt_cuts.filter(lambda cut: cut.id in common_ids)

    src_cuts = src_cuts.sort_like(tgt_cuts)
    # adding source trajectory
    tgt_cuts_dict = {c.id: c for c in tgt_cuts} 
    if args.method == 'concat':
        new_cuts  = combine_cuts(src_cuts, tgt_cuts_dict)
    elif args.method == 'alignments':
        # cuts = src_cuts.map(lambda x: add_src_traj(x, chunk_size_ms=args.chunk_size_ms))
        # adding target trajectory
        cuts = src_cuts.map(lambda x: add_tgt_traj(x, alignment_dict=alignment_text_dict, tgt_cuts=tgt_cuts_dict, chunk_min_duration=args.chunk_min_duration))
        
        new_cuts = combine_chunked_cuts(cuts)
    new_cuts.to_shar(
            args.output_dir,
            shard_size=len(new_cuts),
            shard_offset=shard_id,
            fields={
                'recording': 'wav',
                'target_audio': 'wav',
            }
    )