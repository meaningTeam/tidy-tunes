from functools import lru_cache

import torch

from tidytunes.utils import Audio, chunk_list, collate_audios


def get_dnsmos(
    audio: list[Audio],
    personalized: bool = True,
    device: str = "cpu",
    num_threads: int | None = 8,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Computes DNSMOS (Deep Noise Suppression Mean Opinion Score) for a batch of audio clips.

    Args:
        audio (list[Audio]): List of Audio objects.
        personalized (bool): Whether to use a personalized model (default: True).
        device (str): The device to run the model on (default: "cpu").
        num_threads (int | None): Number of threads to use for ONNX inference (default: 8).
        batch_size (int): Batch size for processing (default: 32).

    Returns:
        torch.Tensor: Tensor containing DNSMOS scores for each input audio clip.
    """
    model = load_dnsmos_model(device, personalized, num_threads)
    mos_scores = []

    for audio_batch in chunk_list(audio, batch_size):
        a, al = collate_audios(audio_batch, model.sampling_rate)
        with torch.no_grad():
            _, _, _, mos = model(a.to(device), al.to(device))
        mos_scores.append(mos)

    return torch.cat(mos_scores, dim=0)


@lru_cache(1)
def load_dnsmos_model(
    device: str = "cpu",
    personalized: bool = True,
    num_threads: int | None = None,
    tag: str = None,
):
    """
    Loads and caches the DNSMOS model for speech quality assessment.

    Args:
        device (str): The device to run the model on (default: "cpu").
        personalized (bool): Whether to use a personalized model (default: True).
        num_threads (int | None): Number of threads to use for ONNX inference (default: None).
        tag (str): Version tag for downloading model weights

    Returns:
        DNSMOSPredictor: Loaded DNSMOS model ready for inference.
    """
    from tidytunes.models import DNSMOSPredictor
    from tidytunes.utils.download import download_github
    from tidytunes.utils.onnx import load_onnx_session

    device = torch.device(device)
    prefix = "p_" if personalized else ""

    dnsmos_onnx_path = download_github(f"dnsmos_{prefix}sig_bak_ovr.onnx", tag)
    p808_onnx_path = download_github("dnsmos_model_v8.onnx", tag)
    dnsmos_sess = load_onnx_session(dnsmos_onnx_path, device, num_threads)
    p808_sess = load_onnx_session(p808_onnx_path, device, num_threads)

    model = DNSMOSPredictor(dnsmos_sess, p808_sess, device, personalized).to(device)
    return model
