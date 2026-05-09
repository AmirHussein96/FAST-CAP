# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


def collate_tensors(stream, pad):
    """
    >>> tensors = [torch.tensor(x) for x in [[1,2,3], [1]]]
    >>> pad = 0
    >>> collate_tensors(tensors, pad)
    tensor([[1, 2, 3],
        [1, 0, 0]])
    """
    assert len(stream) > 0

    length = max(v.size(0) for v in stream)
    n_samples = len(stream)

    collated = stream[0].new_full((n_samples, length), pad)

    for i, v in enumerate(stream):
        collated[i, : v.size(0)] = v

    return collated


def wrap_bos_eos(units, durations, dense, bos, eos):
    assert units.size(0) == durations.size(0) == dense.size(0)
    units = torch.cat([bos, units, eos])
    z = torch.zeros_like(durations[0:1])
    durations = torch.cat([z, durations, z])
    z = torch.zeros_like(dense[0:1, :])
    dense = torch.cat([z, dense, z])

    return units, durations, dense
