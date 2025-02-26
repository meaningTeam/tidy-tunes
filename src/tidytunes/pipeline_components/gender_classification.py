from functools import lru_cache

import torch

from tidytunes.pipeline_components.speaker_segmentation import load_speaker_encoder
from tidytunes.utils import Audio, collate_audios


def is_male(audios: list[Audio], device="cpu"):

    speaker_encoder = load_speaker_encoder(device=device)
    model = load_gender_classification_model()

    with torch.no_grad():
        audio, audio_lens = collate_audios(
            audios, sampling_rate=speaker_encoder.sampling_rate
        )
        audio = audio.to(device)
        audio_lens = audio_lens.to(device)
        embeddings = speaker_encoder(audio, audio_lens)
        embeddings_flattened = [e.mean(dim=0) for e in embeddings]

    classifications = [
        model.predict(e.reshape(1, -1))
        for e in torch.stack(embeddings_flattened).cpu().numpy()
    ]

    return [True if c == 1 else False for c in classifications]


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
