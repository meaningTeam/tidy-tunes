from functools import lru_cache

import torch

from tidytunes.utils import (
    Audio,
    Segment,
    batched,
    collate_audios,
    frame_labels_to_time_segments,
)


def merge_speech_segments(
    segments: list[Segment],
    min_duration: float,
    max_duration: float,
) -> list[Segment]:
    """
    Greedily merges consecutive speech segments (including silence gaps between them)
    into longer segments that respect duration constraints.

    Iterates left-to-right, accumulating adjacent segments into one span as long as the
    total duration (from the first segment's start to the last segment's end) stays
    within max_duration. The merged segment preserves continuity -- silence between
    chunks is included, not cut out. After merging, only segments with
    duration >= min_duration are kept.

    Args:
        segments: Speech segments in chronological order.
        min_duration: Minimum duration for a valid output segment.
        max_duration: Maximum duration for a valid output segment.

    Returns:
        List of merged segments satisfying the duration constraints.
    """
    if not segments:
        return []

    merged: list[Segment] = []
    accumulator_start = segments[0].start
    accumulator_end = segments[0].start + segments[0].duration

    for segment in segments[1:]:
        segment_end = segment.start + segment.duration
        candidate_duration = segment_end - accumulator_start

        if candidate_duration <= max_duration:
            accumulator_end = segment_end
        else:
            merged.append(Segment(
                start=accumulator_start,
                duration=round(accumulator_end - accumulator_start, 3),
                symbol=True,
            ))
            accumulator_start = segment.start
            accumulator_end = segment_end

    merged.append(Segment(
        start=accumulator_start,
        duration=round(accumulator_end - accumulator_start, 3),
        symbol=True,
    ))

    return [segment for segment in merged if segment.duration >= min_duration]


@batched(batch_size=1024, batch_duration=1280.0)
def find_segments_with_speech(
    audio: list[Audio],
    min_duration: float = 3.2,
    max_duration: float = 30.0,
    prebounce_frames: int = 3,
    device: str = "cpu",
):
    """
    Identifies speech segments in the given audio using a Voice Activity Detector (VAD).

    Speech chunks shorter than min_duration are greedily merged with their neighbors
    (preserving continuity including silence gaps) to recover segments that would
    otherwise be discarded.

    Args:
        audio (list[Audio]): List of Audio objects.
        min_duration (float): Minimum duration for a valid speech segment (default: 3.2).
        max_duration (float): Maximum duration for a valid speech segment (default: 30.0).
        prebounce_frames (int): Number of frames (64ms) to shift the speech starts to the left (default: 2).
        device (str): The device to run the VAD model on (default: "cpu").

    Returns:
        list[list[Segment]]: Time segments containing speech for each input Audio.
    """
    vad = load_vad(device)

    audio_tensor, _ = collate_audios(audio, vad.sampling_rate)
    with torch.no_grad():
        speech_mask = vad(audio_tensor.to(device))

    results = []
    for mask, audio_item in zip(speech_mask, audio):
        speech_segments = frame_labels_to_time_segments(
            mask,
            vad.frame_shift,
            filter_with=lambda segment: segment.symbol is True
            and segment.duration <= max_duration,
            segment_duration=audio_item.duration,
            prebounce_frames=prebounce_frames,
        )
        results.append(merge_speech_segments(speech_segments, min_duration, max_duration))

    return results


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
    from tidytunes.models.external import SileroVADv6
    from tidytunes.utils.download import download_github

    model_weights_path = download_github("silero_vad_v6.2.pt", tag)
    vad = SileroVADv6.from_files(model_weights_path)
    vad_trace = vad.to_jit_trace(device)
    return VoiceActivityDetector(vad_trace, vad.frame_shift, vad.sampling_rate).to(
        device
    )
