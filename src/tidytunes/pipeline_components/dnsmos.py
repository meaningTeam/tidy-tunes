from functools import lru_cache

import torch

from tidytunes.utils import Audio, batched, collate_audios


@batched(batch_size=1024, batch_duration=1280.0)
def get_dnsmos(
    audio: list[Audio],
    personalized: bool = True,
    device: str = "cpu",
    num_threads: int | None = 8,
) -> torch.Tensor:
    """
    Computes DNSMOS (Deep Noise Suppression Mean Opinion Score) for a batch of audio clips.

    Args:
        audio (list[Audio]): List of Audio objects.
        personalized (bool): Whether to use a personalized model (default: True).
        device (str): The device to run the model on (default: "cpu").
        num_threads (int | None): Number of threads to use for ONNX inference (default: 8).

    Returns:
        torch.Tensor: Tensor containing DNSMOS scores for each input audio clip.
    """

    model = load_dnsmos_model(device, personalized, num_threads)

    a, al = collate_audios(audio, model.sampling_rate)
    with torch.no_grad():
        _, _, _, mos = model(a.to(device), al.to(device))

    return torch.unbind(mos)


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
