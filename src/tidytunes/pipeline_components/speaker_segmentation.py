from functools import lru_cache

import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans

from tidytunes.utils import (
    Audio,
    batched,
    collate_audios,
    frame_labels_to_time_segments,
)


def find_segments_with_single_speaker(
    audio: list[Audio],
    min_duration: float = 3.2,
    segment_start_shift: float = 0.32,
    segment_end_shift: float = 0.96,
    frame_shift: int = 64,
    num_clusters: int = 10,
    device: str = "cpu",
):
    """
    Identifies segments in the audio where only a single speaker is present.
    *** All input segments are supposed to come from a single source. ***

    Args:
        audio (list[Audio]): List of audio objects.
        min_duration (float): Minimum duration (in seconds) for a valid segment (default: 3.2).
        segment_start_shift (float): Shift (in seconds) for a segment start (default: 0.32).
        segment_end_shift (float): Shift (in seconds) for a segment end (default: 0.96).
        frame_shift (float): Number of model input frames per one output speaker label (default: 64).
        num_clusters (int): Initial number of clusters before agglomertive clustering (defailt: 10).
        device (str): Device to run the model on (default: "cpu").

    Returns:
        list[list[Segment]]: List of speaker segments for each input audio.
    """

    embeddings = get_speaker_embeddings(audio, frame_shift, device)
    embeddings_all = torch.stack(embeddings, dim=0)

    centroids = find_cluster_centers(embeddings_all, num_clusters)
    labels = [
        F.cosine_similarity(e.unsqueeze(1), centroids.unsqueeze(0), dim=-1).argmax(
            dim=-1
        )
        for e in embeddings
    ]

    speaker_encoder = load_speaker_encoder(num_frames=frame_shift, device=device)
    frame_shift_seconds = (
        frame_shift * speaker_encoder.hop_length / speaker_encoder.sampling_rate
    )
    time_segments = [
        frame_labels_to_time_segments(
            l,
            frame_shift=frame_shift_seconds,
            filter_with=lambda x: x.duration >= min_duration,
        )
        for l in labels
    ]

    # Adjust segments to make sure there are no cross-talks on boundaries
    for ts in time_segments:
        if len(ts) > 1:
            for t in ts:
                t.start += segment_start_shift
                t.duration -= segment_end_shift

    return time_segments


@batched(batch_size=1024, batch_duration=1280.0)
def get_speaker_embeddings(
    audio: list[Audio], frame_shift: int = 64, device: str = "cpu"
):
    speaker_encoder = load_speaker_encoder(num_frames=frame_shift, device=device)
    a, al = collate_audios(audio, sampling_rate=speaker_encoder.sampling_rate)
    with torch.no_grad():
        e = speaker_encoder(a.to(device), al.to(device))
    return torch.unbind(e)


def find_cluster_centers(embeddings: torch.Tensor, num_clusters):
    """
    Clusters speaker embeddings and refines cluster centers.

    Args:
        embeddings (N, D): Speaker embeddings.
        num_clusters (int): Initial number of clusters before agglomertive clustering.

    Returns:
        Cluster centers of shape (C, D).
    """
    num_clusters = min(len(embeddings), num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(embeddings.cpu())

    ag = AgglomerativeClustering(
        metric="cosine", n_clusters=None, distance_threshold=0.6, linkage="complete"
    ).fit(kmeans.cluster_centers_)

    centroids = []
    for i in range(ag.labels_.max() + 1):
        c = torch.from_numpy(kmeans.cluster_centers_[ag.labels_ == i]).float()
        c = c.mean(dim=0, keepdim=True)
        c = F.normalize(c, p=2.0, dim=1)
        centroids.append(c)

    centroids = torch.cat(centroids, dim=0)
    centroids = centroids.to(embeddings.device)

    return centroids


@lru_cache(maxsize=1)
def load_speaker_encoder(num_frames: int = 64, device: str = "cpu", tag: str = None):
    """
    Loads the speaker encoder model.

    Args:
        num_frames (int): Number of frames per input sample (default: 64).
        device (str): Device to run the model on (default: "cpu").
        tag (str): Model version tag

    Returns:
        SpeakerEncoder: Pre-trained speaker encoder model.
    """
    from tidytunes.models import SpeakerEncoder
    from tidytunes.models.external import ResNetSpeakerEncoder
    from tidytunes.utils.download import download_github

    model_weights_path = download_github("coqui_speaker_encoder.pt", tag)
    spk_enc = ResNetSpeakerEncoder.from_files(model_weights_path)
    spk_enc.num_input_frames = num_frames
    sampling_rate = spk_enc.sample_rate
    hop_length = spk_enc.hop_length
    spk_enc = spk_enc.to_jit_trace(device)

    spk_enc = SpeakerEncoder(spk_enc, num_frames, hop_length, sampling_rate)
    spk_enc = spk_enc.eval().to(device)

    return spk_enc
