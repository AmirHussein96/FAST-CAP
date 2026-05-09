# converts original LibriTTS filelists to LibriTTS_R version
# filelists are almost the same, but some waveforms are missing in LibriTTS_R so these will be filtered out
import os, glob
import tqdm

DATA_ROOT = "/lustre/fsw/portfolios/adlr/users/sanggill/TEMP_datasets/LibriTTS-R/LibriTTS_R"
FILELIST_OLD_ROOT = os.path.join(DATA_ROOT, "filelists_old")
FILEPATH_NEW_ROOT = os.path.join(DATA_ROOT, "filelists_new")
os.makedirs(FILEPATH_NEW_ROOT, exist_ok=True)

filelists = glob.glob(os.path.join(FILELIST_OLD_ROOT, "*.txt"))

for filepath in filelists:
    print("INFO: processing {}".format(filepath))
    filepath_basename = os.path.basename(filepath)
    filepath_new = os.path.join(FILEPATH_NEW_ROOT, filepath_basename)
    filepath_new_notfound = os.path.join(FILEPATH_NEW_ROOT, "WAVNOTFOUND_"+filepath_basename)
    # scan old filelist and look for errors
    with open(filepath, 'r') as f:
        lines = f.readlines()
    maybe_wavpath = lines[0].split("|")[0]
    f_notfound = None
    if maybe_wavpath.endswith(".wav"):
        f_notfound = open(filepath_new_notfound, 'w')
    with open(filepath_new, 'w') as f:
        for line in tqdm.tqdm(lines):
            maybe_wavpath = line.split("|")[0]
            if maybe_wavpath.endswith(".wav"):
                if not os.path.exists(os.path.join(DATA_ROOT, "24khz", maybe_wavpath)):
                    print("WARNING: file not found {}".format(maybe_wavpath))
                    f_notfound.write(line)
                    continue
                f.write(line)
            else:
                f.write(line)            
    if f_notfound is not None:
        f_notfound.close()