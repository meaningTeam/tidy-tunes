from functools import lru_cache

import torch

from tidytunes.utils import Audio
from tidytunes.utils.memory import garbage_collection_cuda, is_cufft_snafu


def get_rolloff_frequency(
    audio: list[Audio],
    roll_percent: float = 0.995,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the roll-off frequency for a list of audio segments.

    audio (list[Audio]): The list of audio segments.
    roll_percent (float): The roll-off percentage to compute the frequency (default: 0.995).
    device (str): The device to run the computation on (default: "cpu").

    Returns:
        A tensor of shape (B,) containing the roll-off frequencies for each audio segment.
    """
    frequencies = []
    for a in audio:
        w = a.as_tensor().to(device)
        extractor = get_rolloff_extractor(a.sampling_rate, roll_percent, device)

        num_retry = 2
        for attempt in range(num_retry):
            try:
                with torch.no_grad():
                    rolloff = extractor(w)
            except RuntimeError as e:
                if not is_cufft_snafu(e):
                    raise
                if attempt == 0:
                    garbage_collection_cuda()
                else:
                    raise

        frequencies.append(rolloff)
    return torch.stack(frequencies)


@lru_cache()
def get_rolloff_extractor(sampling_rate: int, roll_percent: float, device: str):
    """
    Loads and caches the roll-off frequency extractor model.

    sampling_rate (int): The sampling rate of the audio.
    roll_percent (float): The roll-off percentage.
    device (str): The device to place the module on (default: "cpu").

    Returns:
        A roll-off frequency extractor model.
    """
    from tidytunes.models import RollOff

    if device != "cpu":
        torch.backends.cuda.cufft_plan_cache[0].max_size = 0

    rolloff = RollOff(sampling_rate, roll_percent)
    rolloff = rolloff.to_jit_trace(device)
    return rolloff
