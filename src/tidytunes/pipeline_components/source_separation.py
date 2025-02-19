from functools import lru_cache

import torch

from tidytunes.utils import (
    Audio,
    collate_audios,
    frame_labels_to_time_segments,
    masked_mean,
    sequence_mask,
)


def find_segments_without_music(
    audios: list[Audio],
    frame_shift: float = 0.16,
    max_music_energy: float = 0.01,
    min_speech_energy: float = 0.99,
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

    audio, audio_lens = collate_audios(audios, demucs.sampling_rate)
    audio, audio_lens = audio.to(device), audio_lens.to(device)

    with torch.no_grad():
        sources = demucs(audio, audio_lens)

    B, C, T = sources.shape
    hop_length = int(frame_shift * demucs.sampling_rate)
    window_size = hop_length * 2

    is_speech_without_music = []
    n_frames = (audio_lens - window_size) // hop_length + 1

    # NOTE: Simply unfolding the whole sources can easily cause OOMs for longer
    # inputs, so we rather go slowly frame by frame to be memory efficient
    # TODO: Implement chunk-wise inference instead of frame-wise
    for i in range(T // hop_length):
        start, end = i * hop_length, i * hop_length + window_size
        energy = masked_mean(sources[..., start:end] ** 2)

        mask = sequence_mask(n_frames.clamp(max=1, min=0), max_length=1)
        energy[~mask.expand_as(energy)] = 0.0
        n_frames -= 1

        # Merge all non-vocal channels into a single one
        energy = torch.cat(
            (energy[..., :3].sum(dim=-1, keepdim=True), energy[..., 3:]), dim=-1
        )
        energy /= energy.sum(dim=-1, keepdim=True)

        no_music = energy[..., 0] <= max_music_energy
        is_speech = energy[..., 1] >= min_speech_energy
        is_speech_without_music.append(no_music & is_speech)

    is_speech_without_music = torch.stack(is_speech_without_music, dim=1)

    return [
        frame_labels_to_time_segments(
            frames,
            frame_shift,
            filter_with=lambda x: (x.symbol is True) & (x.duration >= min_duration),
        )
        for frames in is_speech_without_music
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
