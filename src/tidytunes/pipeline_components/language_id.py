from functools import lru_cache

import torch

from tidytunes.utils import Audio, collate_audios, to_batches


def get_language_probabilities(
    audio: list[Audio],
    language_code: str,
    device: str = "cpu",
    batch_size: int = 64,
    batch_duration: float = 1280.0,
) -> torch.Tensor:
    """
    Compute the probability of a given language being spoken in the audio.

    Args:
        audio (list[Audio]): List of Audio objects to analyze.
        language_code (str): The target language code to check probabilities for.
        device (str): The device to run the model on (default: "cpu").
        batch_size (int): Maximal number of audio samples to process in a batch (default: 64).
        batch_duration (float): Maximal duration of audio samples to process in a batch (default: 1280.0)

    Returns:
        Tensor (B,) of probabilities for the specified language.
    """
    model, lab2ind = load_langid_voxlingua107_ecapa(device)
    lang_probs = []

    for audio_batch in to_batches(audio, batch_size, batch_duration):
        audio_16khz, audio_16khz_lens = collate_audios(audio_batch, 16000)
        with torch.no_grad():
            out_prob, _, _ = model(audio_16khz.to(device), audio_16khz_lens.to(device))
        lang_probs.extend([p[lab2ind[language_code]] for p in out_prob])

    return torch.stack(lang_probs)


@lru_cache(1)
def load_langid_voxlingua107_ecapa(device: str = "cpu", tag: str = None):
    """
    Loads, traces, and caches the pre-trained VoxLingua107 ECAPA model for spoken language identification.

    Args:
        device (str): The device to load the model on (default: "cpu").
        tag (str): Github release tag associated with assets to load
    Returns:
        Tuple: A traced model and a dictionary mapping language labels to indices.
    """
    from tidytunes.models import SpokenLanguageIdentificationModel
    from tidytunes.utils.download import download_github

    model = SpokenLanguageIdentificationModel.from_files(
        download_github("lang_id_voxlingua107_ecapa_classifier.pt", tag),
        download_github("lang_id_voxlingua107_ecapa_embedding_model.pt", tag),
        download_github("lang_id_voxlingua107_ecapa_normalizer.pt", tag),
        download_github("lang_id_voxlingua107_ecapa_label_to_language.json", tag),
    )
    model_trace = model.to_jit_trace(device)
    return model_trace, model.lab2ind
