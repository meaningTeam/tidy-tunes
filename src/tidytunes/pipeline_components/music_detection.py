from functools import lru_cache

import torch

from tidytunes.utils import (
    Audio,
    collate_audios,
    masked_max,
    masked_mean,
    sequence_mask,
)


def get_music_probability(
    audio: list[Audio],
    reduction: str = "max",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the background music probability for a list of audio segments.

    audio (list[Audio]): The list of audio segments.
    reduction (str): The reduction operation to apply to the frame-wise music probabilities (default: "max").
    device (str): The device to run the computation on (default: "cpu").

    Returns:
        A tensor of shape (B,) containing the background music probabilities for each audio segment.
    """

    detector = get_music_detector(device)
    a, al = collate_audios(audio, detector.sampling_rate)

    with torch.no_grad():
        frame_probs, lens = detector(a, al)

    m = sequence_mask(lens)

    if reduction == "max":
        return masked_max(frame_probs, m)
    elif reduction == "mean":
        return masked_mean(frame_probs, m)
    else:
        raise ValueError(
            f"Invalid reduction: {reduction}, must be one of 'max', 'mean'"
        )


@lru_cache()
def get_music_detector(device: str):
    """
    Loads and caches the music detection model.

    device (str): The device to place the module on (default: "cpu").

    Returns:
        A music detection model.
    """
    from tidytunes.models import MusicDetectionModel

    detector = MusicDetectionModel()
    detector.load_state_dict(
        torch.load(
            "/mnt/homes/tomiinek/various_scripts/checkpoints/best_model/model.pt"
        )
    )
    detector = detector.eval().to(device)
    return detector
