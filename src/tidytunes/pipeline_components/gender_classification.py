from functools import lru_cache

import torch

from tidytunes.pipeline_components.speaker_segmentation import load_speaker_encoder
from tidytunes.utils import Audio, collate_audios, to_batches


def is_male(
    audio: list[Audio],
    device: str = "cpu",
    batch_size: int = 64,
    batch_duration: float = 1280.0,
):
    """
    Classifies gender of the speaker in the input audios

    Args:
        audio (list[Audio]): List of audio objects.
        device (str): Device to run the model on (default: "cpu").
        batch_size (int): Maximal number of audio samples to process in a batch (default: 64).
        batch_duration (float): Maximal duration of audio samples to process in a batch (default: 1280.0)

    Returns:
        list[bool]: List of booleans for each input audio, True for males, False for females.
    """

    speaker_encoder = load_speaker_encoder(device=device)
    model = load_gender_classification_model()
    embeddings = []

    for audio_batch in to_batches(audio, batch_size, batch_duration):

        a, al = collate_audios(audio_batch, sampling_rate=speaker_encoder.sampling_rate)
        with torch.no_grad():
            be = speaker_encoder(a.to(device), al.to(device))
        embeddings.extend([e.mean(dim=0) for e in be])

    classifications = [
        model.predict(e.cpu().numpy().reshape(1, -1)) for e in embeddings
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
