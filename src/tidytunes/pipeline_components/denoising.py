from functools import lru_cache

import torch
from pesq import PesqError, pesq

from tidytunes.utils import (
    Audio,
    batched,
    collate_audios,
    decollate_audios,
    sequence_mask,
    to_batches,
)


@batched(batch_size=1024, batch_duration=1280.0)
def denoise(
    audio: list[Audio],
    device: str = "cpu",
) -> list[Audio]:
    """
    Apply denoising to a list of audio samples using a pre-trained model.

    Args:
        audio (list[Audio]): List of audio objects to be denoised.
        device (str): The device to run the denoising model on (default: "cpu").

    Returns:
        list[Audio]: List of denoised audio objects.
    """
    denoiser = load_denoiser(device)

    audio_tensor, audio_lengths = collate_audios(audio, denoiser.sampling_rate)
    mask = sequence_mask(audio_lengths.to(device))
    with torch.no_grad():
        denoised_audio = denoiser(audio_tensor.to(device), mask)
    return decollate_audios(
        denoised_audio,
        audio_lengths,
        denoiser.sampling_rate,
        origin_like=audio,
    )


def get_denoised_pesq(
    audio: list[Audio],
    sampling_rate: int = 16000,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the Perceptual Evaluation of Speech Quality (PESQ) score between original and denoised audio.

    Args:
        audio (list[Audio]): List of audio objects to be denoised.
        sampling_rate (int): The target sampling rate for PESQ computation (default: 16000 Hz).
        device (str): The device to run the denoising model on (default: "cpu").

    Returns:
        torch.Tensor: Tensor containing PESQ scores for each input Audio.
    """
    denoised = denoise(audio, device)
    return [
        pesq(
            sampling_rate,
            ref.resample(sampling_rate).as_tensor().cpu().numpy(),
            enh.resample(sampling_rate).as_tensor().cpu().numpy(),
            on_error=PesqError.RETURN_VALUES,
        )
        for ref, enh in zip(audio, denoised)
    ]


@lru_cache(maxsize=1)
def load_denoiser(device: str = "cpu", tag: str = None):
    """
    Load and cache the pre-trained denoiser model.

    Args:
        device (str): The device to load the model onto (default: "cpu").
        tag (str): Github release tag associated with assets to load

    Returns:
        AttenuateDenoiser: The loaded denoiser model.
    """
    from tidytunes.models.external import AttenuateDenoiser
    from tidytunes.utils.download import download_github

    model_weights_path = download_github("attenuate_weights.pt", tag)
    model = AttenuateDenoiser.from_files(model_weights_path)
    model = model.eval().to(device)

    return model
