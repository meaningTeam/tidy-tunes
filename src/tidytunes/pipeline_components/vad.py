from functools import lru_cache

import torch

from tidytunes.utils import (
    Audio,
    collate_audios,
    frame_labels_to_time_segments,
    to_batches,
)


def find_segments_with_speech(
    audio: list[Audio],
    min_duration: float = 3.2,
    max_duration: float = 30.0,
    device: str = "cpu",
    batch_size: int = 64,
    batch_duration: float = 1280.0,
):
    """
    Identifies speech segments in the given audio using a Voice Activity Detector (VAD).

    Args:
        audio (list[Audio]): List of Audio objects.
        min_duration (float): Minimum duration for a valid speech segment (default: 3.2).
        max_duration (float): Maximum duration for a valid speech segment (default: 30.0).
        device (str): The device to run the VAD model on (default: "cpu").
        batch_size (int): Maximal number of audio samples to process in a batch (default: 64).
        batch_duration (float): Maximal duration of audio samples to process in a batch (default: 1280.0)

    Returns:
        list[list[Segment]]: Time segments containing speech for each input Audio.
    """
    vad = load_vad(device)
    time_segments = []

    for audio_batch in to_batches(audio, batch_size, batch_duration):

        audio_tensor, _ = collate_audios(audio_batch, vad.sampling_rate)
        with torch.no_grad():
            speech_mask = vad(audio_tensor.to(device))
        speech_mask[..., :-1] += speech_mask[
            ..., 1:
        ].clone()  # Pre-bounce speech starts

        time_segments.extend(
            [
                frame_labels_to_time_segments(
                    m,
                    vad.frame_shift,
                    filter_with=lambda x: (x.symbol is True)
                    and (min_duration <= x.duration <= max_duration),
                )
                for m in speech_mask
            ]
        )

    return time_segments


@lru_cache(maxsize=1)
def load_vad(device: str = "cpu", tag: str = "v1.0.0"):
    """
    Loads, traces, and caches the Voice Activity Detector (VAD) model.

    Args:
        device (str): The device to run the VAD model on (default: "cpu").
        tag (str): The version tag for downloading the model (default: "v1.0.0").

    Returns:
        VoiceActivityDetector: Loaded VAD model.
    """
    from tidytunes.models import VoiceActivityDetector
    from tidytunes.models.external import SileroVAD
    from tidytunes.utils.download import download_github

    model_weights_path = download_github(tag, "silerovad_weights.pt")
    vad = SileroVAD.from_files(model_weights_path)
    vad_trace = vad.to_jit_trace(device)
    return VoiceActivityDetector(vad_trace).to(device)
