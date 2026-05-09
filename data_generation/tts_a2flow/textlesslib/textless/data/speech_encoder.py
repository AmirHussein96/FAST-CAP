# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn
import torchaudio
from typing import Optional, Callable, Dict
from textlesslib.textless import dispatch_dense_model, dispatch_quantizer
from textlesslib.textless import dispatch_dense_model
from .collater_utils import wrap_bos_eos, collate_tensors


def get_streams(
    waveform,
    speaker,
    dense_model,
    quantizer_model,
    deduplicate,
):
    # if waveform.ndim > 1:
    #     waveform = waveform.mean(0)

    dense_features = dense_model(waveform)
    units = quantizer_model(dense_features)

    if deduplicate:
        units_list, durations_list = [], []
        B = units.size(0)
        for i in range(B):
            units_i, durations_i = torch.unique_consecutive(units[i], return_counts=True)
            units_list.append(units_i)
            durations_list.append(durations_i)
        return units_list, durations_list, dense_features
    else:
        durations = torch.ones_like(units)
        return units, durations, dense_features


class SpeechEncoder(torch.nn.Module):
    """SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations.
    """

    def __init__(
        self,
        dense_model: torch.nn.Module,
        quantizer_model: torch.nn.Module,
        deduplicate: bool,
    ):
        """Builds a SpeechEncoder instance. SpeechEncoder encodes speech into streams of (pseudo-)units, unit durations.

        Args:
            dense_model (torch.nn.Module): Dense module used to represent the audio
            quantizer_model (torch.nn.Module): Quantize module that converts dense representation into discrete tokens
            deduplicate (bool): if set, run-length encoding is applied so that repeated tokens are deduplicated
                and duration channel contains the number of repeats of the token.
        """
        super().__init__()
        self.dense_model = dense_model
        self.quantizer_model = quantizer_model

        self.deduplicate = deduplicate

        self.unit_vocab_size = self.quantizer_model.vocab_size

        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @classmethod
    def by_name(
        cls,
        dense_model_name: str,
        quantizer_model_name: str,
        vocab_size: int,
        deduplicate: bool,
    ) -> "SpeechEncoder":
        """Builds a SpeechEncoder instance by retrieving pre-trained dense and quantizer models specified by their parameters
        (names and vocabulary size).

        Args:
            dense_model_name (str): Name of the dense module used to represent the audio
            quantizer_model_name (str): Name of the quantizer module that converts dense representation into discrete tokens
            vocab_size (int): Specifies the codebook size
            deduplicate (bool): if set, run-length encoding is applied so that repeated tokens are deduplicated
                and duration channel contains the number of repeats of the token.
        """
        dense_model = dispatch_dense_model(dense_model_name)
        quantizer_model = dispatch_quantizer(
            dense_model_name, quantizer_model_name, vocab_size
        )

        return cls(
            dense_model,
            quantizer_model,
            deduplicate,
        )

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: device where the speech encoder resides
        """
        return self._float_tensor.device

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: vocabulary size used for the unit stream (NB: not counting bos/eos/pad tokens)
        """
        return self.quantizer_model.vocab_size

    @property
    def code_hop_size(self) -> int:
        """
        Returns:
            int: hop step size of the dense model
        """
        return self.dense_model.code_hop_size

    @property
    def expected_sample_rate(self) -> int:
        """
        int: sample rate expected by the underlying dense model
        """
        return self.dense_model.expected_sample_rate

    def maybe_resample(
        self, waveform: torch.Tensor, input_sample_rate: int
    ) -> torch.Tensor:
        """
        Takes a waveform and input rate and resamples it into the
        sample rate expected by the encoder (and underlying dense model). Does nothing
        if the sample rates coincide.
        Args:
            waveform (torch.Tensor): audio stream
            input_sample_rate (int): sample rate of the original audio

        Returns:
            torch.Tensor: audio, potentially resampled to match the expected
            sample rate of the encoder
        """
        if input_sample_rate == self.expected_sample_rate:
            return waveform
        return torchaudio.functional.resample(
            waveform, input_sample_rate, self.expected_sample_rate
        )

    def forward(
        self, waveform: torch.Tensor, speaker: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Encodes a raw waveform tensor into two or three aligned & synchronised streams: pseudo-unit (token),
        duration.

        Args:
            waveform (torch.Tensor): audio to be encoded

        Returns:
            Dict[str, torch.Tensor]: dictionary with the following keys:
             * "units": contains an int tensor with the unit stream,
             * "durations": duration of each unit, measured in frames,
             * "dense": dense encoding of the audio, as provided by the underlying dense model,
        """
        units, durations, dense_features = get_streams(
            waveform,
            speaker,
            self.dense_model,
            self.quantizer_model,
            self.deduplicate,
        )
        if self.deduplicate:
            units = collate_tensors(units, pad=self.unit_vocab_size)
            durations = collate_tensors(durations, pad=0)

        item = {
            "units": units.to(self.device),
            "durations": durations.to(self.device),
            "dense": dense_features,
        }
        return item
