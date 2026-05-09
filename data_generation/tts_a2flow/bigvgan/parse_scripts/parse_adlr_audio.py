import os
import soundfile as sf
from soundfile import LibsndfileError
import argparse
from tqdm import tqdm
import concurrent.futures

def check_output_files_exist_and_not_empty(output_path_root, subdir):
    """
    Check if the main audio filelist exists and is not empty.
    The error and bad quality filelists can be empty if all files are good.
    """
    # Main file list that must exist and not be empty
    main_file_path = os.path.join(output_path_root, subdir + "_audio_filelist.txt")

    # Check if the main file list exists and is not empty
    if not os.path.exists(main_file_path) or os.stat(main_file_path).st_size == 0:
        return False

    # If the main file list exists and is not empty, the directory is considered processed
    return True


def flush_output_files(output_path_root, subdir):
    """Flush the write operations for output files."""
    # Paths for the output files
    output_files = [
        os.path.join(output_path_root, subdir + suffix)
        for suffix in ["_audio_filelist.txt", "_error_filelist.txt", "_badquality_filelist.txt"]
    ]
    
    for file_path in output_files:
        with open(file_path, 'a') as file:
            file.flush()

def write_file_info(f, file_path, subdir_root, duration):
    relative_path = os.path.relpath(file_path, subdir_root)
    # Ensure thread safety when writing to file
    f.write(f"{relative_path}|DUMMY|DUMMY|other|{duration}\n")
    
def process_outputs(outputs, output_path_root, subdir):
    output_path = os.path.join(output_path_root, subdir + "_audio_filelist.txt")
    error_path = os.path.join(output_path_root, subdir + "_error_filelist.txt")
    bad_quality_path = os.path.join(output_path_root, subdir + "_badquality_filelist.txt")

    with open(output_path, 'w') as f, open(error_path, 'w') as f_error, open(bad_quality_path, 'w') as f_badquality:
        for output in outputs:
            file_path, data_subdir_root, duration, status = output
            if status == "good":
                write_file_info(f, file_path, data_subdir_root, duration)
            elif status == "bad_quality":
                write_file_info(f_badquality, file_path, data_subdir_root, duration)
            elif status == "error":
                write_file_info(f_error, file_path, data_subdir_root, duration)
    
def process_audio_file(
    file_path,
    list_audio_extension,
    min_sampling_rate,
    output_path_root,
    subdir,
    data_subdir_root,
    require_stereo
):
    if not file_path.lower().endswith(tuple(list_audio_extension)):
        return  # Skip files with extensions not in the list
    
    output = None
    try:
        with sf.SoundFile(file_path) as track:
            duration = float(track.frames) / track.samplerate
            # Check for minimum frame count, sampling rate, and optionally if stereo is required
            bad_quality_criteria = track.frames < 8192 or \
                track.samplerate < min_sampling_rate or \
                (require_stereo and track.channels != 2)
            if bad_quality_criteria:
                output = (file_path, data_subdir_root, duration, "bad_quality")
            else: # passed all criteria
                output = (file_path, data_subdir_root, duration, "good")
    except LibsndfileError as e:
        print(f"Error thrown for {file_path}: {e}")
        output = (file_path, data_subdir_root, "0.0", "error")

    return output

# def process_audio_files(subdir, data_root, output_path_root, list_audio_extension, min_sampling_rate):
#     # Skip processing if all output files exist and are not empty
#     if check_output_files_exist_and_not_empty(output_path_root, subdir):
#         print(f"Skipping {subdir} as output files already exist and are not empty.")
#         return
    
#     data_subdir_root = os.path.join(data_root, subdir)
#     print(f"gathering all files from {data_subdir_root}")
#     all_files = [os.path.join(root, file) for root, _, files in os.walk(data_subdir_root) for file in files]
    
#     outputs = []
#     print(f"launching parallel processing")
#     with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
#         futures = {executor.submit(process_audio_file, file_path, list_audio_extension, min_sampling_rate, output_path_root, subdir, data_subdir_root): file_path for file_path in all_files}
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing files in {subdir}"):
#             output = future.result()
#             if output:
#                 outputs.append(output)
                
#     if outputs == []: # subdir contains no valid data
#         print(f"{subdir} does not contain valid data. skipping making output filelits")
#         return
    
#     print(f"writing and flushing outputs to {output_path_root}")
#     process_outputs(outputs, output_path_root, subdir)
#     # Flush files after processing
#     flush_output_files(output_path_root, subdir)

def process_audio_files(subdir, data_root, output_path_root, list_audio_extension, min_sampling_rate, require_stereo):
    # Skip processing if all output files exist and are not empty
    if check_output_files_exist_and_not_empty(output_path_root, subdir):
        print(f"Skipping {subdir} as output files already exist and are not empty.")
        return
    
    data_subdir_root = os.path.join(data_root, subdir)
    print(f"gathering all files from {data_subdir_root}")
    all_files = [os.path.join(root, file) for root, _, files in os.walk(data_subdir_root) for file in files]
    
    outputs = []
    print(f"processing files in {subdir}")
    for file_path in tqdm(all_files, desc=f"Processing files in {subdir}"):
        output = process_audio_file(file_path, list_audio_extension, min_sampling_rate, output_path_root, subdir, data_subdir_root, require_stereo)
        if output:
            outputs.append(output)

    if outputs == []: # subdir contains no valid data
        print(f"{subdir} does not contain valid data. Skipping making output filelists.")
        return
    
    print(f"writing and flushing outputs to {output_path_root}")
    process_outputs(outputs, output_path_root, subdir)
    # Flush files after processing
    flush_output_files(output_path_root, subdir)
    
def print_args(args):
    print("Running with the following configurations:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

def main(args):
    os.makedirs(args.output_path_root, exist_ok=True)
    all_subdirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]

    for directory in tqdm(all_subdirs, desc="Processing directories"):
        # if directory in ["AudioLM_datasets", "models", "karan", "llama", "makeanaudio2_dataset", "metadata", "music-datasets", "syn-cap-for-TTA-checkpoints"]:
        #     print(f"skipping {directory}: process {directory} separately by providing this explicitely as --data_root")
        #     continue
        print(f"processing {directory}")
        process_audio_files(directory, args.data_root, args.output_path_root, args.list_audio_extension, args.min_sampling_rate, args.require_stereo)

    print("Done processing audio files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/AudioLM_datasets", help='Root directory of the data')
    parser.add_argument('--output_path_root', type=str, default="/lustre/fsw/portfolios/adlr/users/sanggill/filelists/adlr_audio_music_datasets_44khz_stereo", help='Root directory for output file lists')
    parser.add_argument('--list_audio_extension', nargs='+', default=['.wav', '.mp3', '.flac', '.ogg'], help='List of audio file extensions to include')
    parser.add_argument('--min_sampling_rate', type=int, default=44100, help='Minimum acceptable sampling rate for audio files in Hz')
    parser.add_argument('--require_stereo', default=False, action='store_true', help="only accept stereo audio (track.channels == 2) to be included to the filelist")
    args = parser.parse_args()

    print_args(args)
    main(args)
