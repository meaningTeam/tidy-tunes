from functools import lru_cache

import torch

from tidytunes.pipeline_components.speaker_segmentation import load_speaker_encoder
from tidytunes.utils import Audio, batched, collate_audios


@batched(batch_size=1024, batch_duration=1280.0)
def is_male(
    audio: list[Audio],
    device: str = "cpu",
):
    """
    Classifies gender of the speaker in the input audios

    Args:
        audio (list[Audio]): List of audio objects.
        device (str): Device to run the model on (default: "cpu").

    Returns:
        list[bool]: List of booleans for each input audio, True for males, False for females.
    """

    speaker_encoder = load_speaker_encoder(device=device)
    model = load_gender_classification_model()
    embeddings = []

    a, al = collate_audios(audio, sampling_rate=speaker_encoder.sampling_rate)
    with torch.no_grad():
        embeddings = speaker_encoder(a.to(device), al.to(device))
    classifications = [
        model.predict(e.mean(dim=0).cpu().numpy().reshape(1, -1)) for e in embeddings
    ]
    return [c == 1 for c in classifications]


@lru_cache(maxsize=1)
def load_gender_classification_model(tag: str = None):
    """
    Loads the gender classification model.

    Args:
        tag (str): Model version tag

    Returns:
        KMeans: Pre-trained gender classification encoder model.
    """
    import joblib

    from tidytunes.utils.download import download_github

    classificator = download_github("gender_recognition_model.pkl", tag)
    with open(classificator, "rb") as f:
        model = joblib.load(f)
    return model
