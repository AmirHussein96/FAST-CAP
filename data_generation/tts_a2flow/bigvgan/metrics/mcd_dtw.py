from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional
from typing import OrderedDict as ODType
from typing import Tuple, cast

import numpy as np
import pandas as pd
from librosa import load
from librosa.feature import melspectrogram
from pandas import DataFrame
from tqdm import tqdm
import pysptk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import librosa

from mel_cepstral_distance.core import (get_mcd_and_penalty_and_final_frame_number,
                                        get_mfccs_of_mel_spectrogram)
from mel_cepstral_distance.helper import get_all_files_in_all_subfolders
from mel_cepstral_distance.mcd_computation import get_metrics_mels
from mel_cepstral_distance.types import Frames, MelCepstralDistance, Penalty

###########################################################
# "v1" mcd impl used from bigvgan paper and public bigvsan
# this is used for matching the number of PwC mostly
def readmgc(x):
    frame_length = 1024
    hop_length = 256
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    return mgc

def calculate_mcd_v1(y_true, y_pred, sr):
    """
    Calculate the Mel-Cepstral Distortion (MCD) between the true and predicted audio.

    Parameters:
    y_true (np.ndarray): The ground truth audio signal.
    y_pred (np.ndarray): The predicted audio signal.
    sr (int): MUST BE 22050 for this v1 impl
    Returns:
    float: The MCD value.
    """
    assert sr == 22050, "calculate_mcd_v1 MUST be sr==22050 and audio inputs MUST be resampled to this sr. check again!"
    
    y_double = y_true.astype(np.float64)
    y_g_hat_double = y_pred.astype(np.float64)

    y_mgc = readmgc(y_double)
    y_g_hat_mgc = readmgc(y_g_hat_double)

    _, path = fastdtw(y_mgc, y_g_hat_mgc, dist=euclidean)

    y_path = list(map(lambda l: l[0], path))
    y_g_hat_path = list(map(lambda l: l[1], path))
    y_mgc = y_mgc[y_path]
    y_g_hat_mgc = y_g_hat_mgc[y_g_hat_path]

    frames_tot = y_mgc.shape[0]
    z = y_mgc - y_g_hat_mgc
    s = np.sqrt((z * z).sum(-1)).sum()

    mcd = 10.0 / np.log(10.0) * np.sqrt(2.0) * float(s) / float(frames_tot)
    return mcd
########## end of mcd v1###################

###########################################################
# mcd v2 with modern impl
def get_metrics_wavs_from_numpy(audio_1: np.ndarray, audio_2: np.ndarray, sr: int, *, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True, norm: Optional[Any] = None, dtype: np.dtype = np.float64, n_mfcc: int = 16, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """
  Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

  Parameters
  ----------
  audio_1 : np.ndarray
  audio_2: np.ndarray

  hop_length : int > 0 [scalar]
      specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
      plays a role in computing the (mel-)spectrograms which are needed to compute the mel-cepstral coefficients
      See `librosa.core.stft`

  n_fft     : int > 0 [scalar]
      `n_fft/2+1` is the number of rows of the spectrograms. `n_fft` should be a power of two to optimize the speed of
      the Fast Fourier Transformation

  window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a vector or array of length `n_fft`

      See `librosa.filters.get_window`

  center    : bool [scalar]
      - If `True`, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier
        transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
      - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

  n_mels    : int > 0 [scalar]
      number of Mel bands to generate

  htk       : bool [scalar]
      use HTK formula instead of Slaney when creating the mel-filter bank

  norm      : {None, 1, np.inf} [scalar]
      determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of
      the mel band (area normalization).  Otherwise, leave all the triangles aiming for a peak value of 1.0

  dtype     : np.dtype
      data type of the output

  n_mfcc    : int > 0 [scalar]
      the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
      zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
      according to Robert F. Kubichek)

  use_dtw  : bool [scalar]
      to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
      `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
      otherwise the array with less columns is filled with zeros from the right side.

  Returns
  -------
  mcd        : float
      the mel-cepstral distance between the two input audios
  penalty    : float
      a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
      with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
      coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
      each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
      reached if no columns were added to either array.
  final_frame_number : int
      the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
      filling up with zeros

  Example
  --------
  Comparing two audios to another audio using the sum of the mel-cepstral distance and the penalty
  >>> mcd_12, penalty_12, _ = get_metrics_wavs_from_numpy(audio_1, audio_2, sr=24000)
  """
  
  # assuning both audios are in same sr. note that this will be very wrong if each audio is different in sr
  sr_1 = sr
  sr_2 = sr
  
  mel_spectrogram1 = melspectrogram(
    y=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  mel_spectrogram2 = melspectrogram(
    y=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  return get_metrics_mels(mel_spectrogram1, mel_spectrogram2, n_mfcc=n_mfcc, take_log=True, use_dtw=use_dtw)