from functools import lru_cache

import torch

from tidytunes.utils import Audio, collate_audios, frame_labels_to_time_segments


def find_segments_with_speech(
    audio: list[Audio],
    min_duration: float = 3.2,
    max_duration: float = 30.0,
    device: str = "cpu",
):
    """
    Identifies speech segments in the given audio using a Voice Activity Detector (VAD).

    Args:
        audio (list[Audio]): List of Audio objects.
        min_duration (float): Minimum duration for a valid speech segment (default: 3.2).
        max_duration (float): Maximum duration for a valid speech segment (default: 30.0).
        device (str): The device to run the VAD model on (default: "cpu").

    Returns:
        list[list[Segment]]: Time segments containing speech for each input Audio.
    """
    vad = load_vad(device)
    audio_tensor, _ = collate_audios(audio, vad.sampling_rate)
    with torch.no_grad():
        speech_mask = vad(audio_tensor.to(device))
    speech_mask[..., :-1] += speech_mask[..., 1:].clone()  # Pre-bounce speech starts

    time_segments = [
        frame_labels_to_time_segments(
            m,
            vad.frame_shift,
            filter_with=lambda x: (x.symbol is True)
            and (min_duration <= x.duration <= max_duration),
        )
        for m in speech_mask
    ]
    return time_segments


@lru_cache(maxsize=1)
def load_vad(device: str = "cpu", tag: str = None):
    """
    Loads, traces, and caches the Voice Activity Detector (VAD) model.

    Args:
        device (str): The device to run the VAD model on (default: "cpu").
        tag (str): The version tag for downloading the model
    Returns:
        VoiceActivityDetector: Loaded VAD model.
    """
    from tidytunes.models import VoiceActivityDetector
    from tidytunes.models.external import SileroVAD
    from tidytunes.utils.download import download_github

    model_weights_path = download_github("silerovad_weights.pt", tag)
    vad = SileroVAD.from_files(model_weights_path)
    vad_trace = vad.to_jit_trace(device)
    return VoiceActivityDetector(vad_trace).to(device)
