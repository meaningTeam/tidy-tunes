from functools import lru_cache

import torch

from tidytunes.utils import (
    Audio,
    batched,
    collate_audios,
    frame_labels_to_time_segments,
)


@batched(batch_size=1024, batch_duration=1280.0)
def find_segments_without_music(
    audio: list[Audio],
    min_duration: float = 6.4,
    device: str = "cpu",
):
    """
    Identifies segments in audio where speech is present but music is absent.

    Args:
        audio (list[Audio]): List of Audio objects.
        frame_shift (float): Time step between frames in seconds (default: 0.16).
        max_music_energy (float): Maximum allowed energy for non-vocal sources to be considered music-free (default: 0.01).
        min_speech_energy (float): Minimum required energy for vocal sources to be considered speech (default: 0.99).
        min_duration (float): Minimum duration (in seconds) for valid speech segments (default: 6.4).
        device (str): The device to run the model on (default: "cpu").

    Returns:
        list[list[Segment]]: List of speech segments without music for each input Audio.
    """
    demucs = load_demucs(device)

    a, al = collate_audios(audio, demucs.sampling_rate)
    with torch.no_grad():
        speech_without_music_mask = demucs(a.to(device), al.to(device))

    return [
        frame_labels_to_time_segments(
            m,
            demucs.frame_shift,
            filter_with=lambda x: (x.symbol is True) & (x.duration >= min_duration),
        )
        for m in speech_without_music_mask
    ]


@lru_cache(1)
def load_demucs(device: str = "cpu"):
    """
    Loads and caches the Demucs source separation model from torchaudio.

    Args:
        device (str): The device to run the model on (default: "cpu").

    Returns:
        SourceSeparator: Loaded Demucs model ready for inference.
    """
    import torchaudio

    from tidytunes.models import SourceSeparator

    pipeline = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    demucs = pipeline._model_factory_func()
    model_path = torchaudio.utils.download_asset(pipeline._model_path)

    state_dict = torch.load(model_path, weights_only=True)
    demucs.load_state_dict(state_dict)

    model = SourceSeparator(demucs).eval().to(device)
    return model
