from functools import lru_cache

import torch

from tidytunes.models import ASRModel
from tidytunes.utils import Audio, batched, collate_audios, compute_wer


@batched(batch_size=128, batch_duration=1280.0)
def get_asr_agreement(
    audio: list[Audio],
    language: str | None = None,
    device: str = "cpu",
    primary_model: str = "whisper",
    secondary_model: str = "voxtral",
) -> list[tuple[float, dict]]:
    """
    Compute ASR agreement (as WER) between two ASR models for audio segments.

    Runs two ASR models on each audio segment, computes WER between their
    transcriptions, and returns the WER values together with annotation dicts
    containing both transcripts. Also associates the transcript from the primary
    model with each audio segment's origin metadata for later saving.

    Args:
        audio (list[Audio]): List of audio segments to process.
        language (str, optional): Language code (e.g., 'en', 'fr') to guide ASR.
        device (str): Device to run the models on (default: "cpu").
        primary_model (str): Name of the primary ASR model. The transcript from
            this model is associated with the audio. Options: "whisper",
            "voxtral". Default: "whisper".
        secondary_model (str): Name of the secondary ASR model used for comparison.
            Options: "whisper", "voxtral". Default: "voxtral".

    Returns:
        list[tuple[float, dict]]: Per-segment tuples of (wer, annotation_dict).
            The annotation dict contains "wer", "primary_transcript", and
            "secondary_transcript".
    """

    a, _ = collate_audios(audio, sampling_rate=16000)
    a = a.to(device)

    primary_model = load_asr_model(primary_model, device)
    secondary_model = load_asr_model(secondary_model, device)

    with torch.no_grad():
        primary_results = primary_model(a, language, return_timestamps=False)
        secondary_results = secondary_model(a, language, return_timestamps=False)

    results = []
    for audio_seg, primary, secondary in zip(audio, primary_results, secondary_results):
        wer = compute_wer(primary.text, secondary.text)
        if audio_seg.origin is not None:
            audio_seg.origin.transcript = primary.text
        annotation = {
            "wer": wer,
            "primary_transcript": primary.text,
            "secondary_transcript": secondary.text,
        }
        results.append((wer, annotation))

    return results


@batched(batch_size=128, batch_duration=1280.0)
def get_asr_transcript(
    audio: list[Audio],
    model: str = "whisper",
    language: str | None = None,
    device: str = "cpu",
) -> list[str]:
    """
    Transcribe audio segments using a single ASR model.

    This is an annotation-only component — it produces transcripts but has no
    meaningful scalar for filtering. The transcript is also stored in
    ``origin.transcript`` for backward-compatible ``.txt`` sidecar saving.

    Args:
        audio (list[Audio]): List of audio segments to transcribe.
        model (str): ASR model name ("whisper" or "voxtral"). Default: "whisper".
        language (str, optional): Language code (e.g., 'en', 'fr') to guide ASR.
        device (str): Device to run the model on (default: "cpu").

    Returns:
        list[str]: Transcript text for each audio segment.
    """
    a, _ = collate_audios(audio, sampling_rate=16000)
    a = a.to(device)
    asr = load_asr_model(model, device)

    with torch.no_grad():
        asr_results = asr(a, language, return_timestamps=False)

    transcripts = []
    for audio_seg, result in zip(audio, asr_results):
        if audio_seg.origin is not None:
            audio_seg.origin.transcript = result.text
        transcripts.append(result.text)

    return transcripts


@lru_cache(maxsize=2)
def load_asr_model(model_name: str, device: str = "cpu") -> ASRModel:
    from tidytunes.models import VoxtralASR, WhisperASR

    if model_name == "whisper":
        model = WhisperASR()
    elif model_name == "voxtral":
        model = VoxtralASR()
    else:
        raise ValueError(
            f"Unknown ASR model: {model_name}. Choose 'whisper' or 'voxtral'."
        )

    model = model.eval().to(device)
    return model
