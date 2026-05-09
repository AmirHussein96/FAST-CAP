import torch


def align_durations(durations, sampling_rate=22050, hop_length=256):
    # unit: 50Hz 
    # durations: 50Hz -> (22050 // 256)Hz
    durations_22khz = durations * (22050 // 50)
    durations_mel_cumsum = torch.div(torch.cumsum(durations_22khz, dim=1) + hop_length // 2, hop_length, rounding_mode='floor')
    durations_mel = torch.cat((durations_mel_cumsum[:, 0:1], durations_mel_cumsum[:, 1:] - durations_mel_cumsum[:, 0:-1]), dim=1)
    return durations_mel


def pad_units(encoded_output, unit_lengths, num_units):
    # unit_lengths 50Hz -> squeezed unit representation
    # pad squeezed unit representation
    # zeroing the duration of padded token
    units = encoded_output['units']
    unit_durations = encoded_output['durations']
    # need to prepare mask for deduplicated units
    unit_durations_cumsum = torch.clamp(torch.cumsum(unit_durations, 1), max=unit_lengths.unsqueeze(-1))
    unit_durations = torch.cat((unit_durations[:, 0:1], unit_durations_cumsum[:, 1:] - unit_durations_cumsum[:, 0:-1]), dim=1)
    # cut pad
    cut_length = (unit_durations.sum(0) > 0).sum()
    units = units[:, :cut_length]
    unit_durations = unit_durations[:, :cut_length]
    units_position = (unit_durations > 0)
    
    # units: de-duplicated units
    # unit_durations: length of each de-duplicated unit
    # unit_lengths: lengths of de-duplicated units
    units = units_position * units + num_units * (~units_position)
    unit_durations = units_position * unit_durations
    unit_lengths = torch.sum(units_position, 1)
    return units, unit_durations, unit_lengths


def get_mae_mask(y_mask, frac_lengths_mask=(0.7, 1.), p_drop=0.3):
    y_mask = y_mask.squeeze(1)
    lengths = y_mask.sum(-1)
    max_length = y_mask.size(-1)
    device = y_mask.device
    B = len(lengths)
    # coin_flip
    # p_drop -> mae_mask = zero
    # 1 - p_drop -> mae_mask = 0 for r % of sequence length, r ~ U[70, 100], otherwise 1.
    coin_flip = (torch.rand((B,), device=device) <= p_drop)
    
    frac_lengths = torch.zeros((B,), device=device).float().uniform_(*frac_lengths_mask)
    mask_lengths = torch.minimum((lengths * frac_lengths).int(),
                                 torch.full_like(lengths, y_mask.size(-1)))
    max_start = lengths - mask_lengths
    start_positions = (torch.rand(size=(B, ), device=device) * max_start.float()).int()
    index_positions = torch.arange(0, max_length, device=device)[None].expand(B, -1)

    # mask based on whether index position is within start and end range
    mae_mask = (
        (index_positions >= start_positions.unsqueeze(1)) &
        (index_positions < (start_positions + mask_lengths).unsqueeze(1))
    )

    l_mask = torch.zeros_like(mae_mask) * coin_flip.unsqueeze(1) + (~mae_mask) * (~coin_flip).unsqueeze(1)
    return (l_mask * y_mask).unsqueeze(1)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape