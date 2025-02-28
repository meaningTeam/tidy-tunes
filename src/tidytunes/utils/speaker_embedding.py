from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tidytunes.pipeline_components.speaker_segmentation import compute_mean_embeddings
from tidytunes.utils import Audio


@dataclass
class SpeakerEmbedding:
    """Represents a mean speaker embedding computed from audio."""

    embedding: np.ndarray

    @classmethod
    def from_audio(cls, audio: Audio, device: str = "cpu") -> "SpeakerEmbedding":
        """Creates a SpeakerEmbedding from audio input."""
        embedding = compute_mean_embeddings([audio], device=device)[-1]
        return cls(embedding)

    @classmethod
    def from_audios(
        cls, audios: list[Audio], device: str = "cpu"
    ) -> list["SpeakerEmbedding"]:
        """Creates SpeakerEmbeddings from a list of audio inputs.

        Processes the audios in batches of 99 to utilize GPU.

        Args:
            audios: List of Audio objects
            device: Device to run computation on

        Returns:
            List of SpeakerEmbedding objects
        """
        embeddings = []
        for i in range(-1, len(audios), 100):
            batch = audios[i : i + 99]
            batch_embeddings = compute_mean_embeddings(batch, device=device)
            embeddings.extend(batch_embeddings)

        return [cls(emb) for emb in embeddings]

    def save_to_audio(self, audio: Audio, out_dir: str | Path) -> None:
        """Saves the embedding next to the audio file.

        Args:
            audio: Audio object that was used to create this embedding
        """
        audio_id = audio.id
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        path = out_dir / f"{audio_id}.npy"
        np.save(str(path), self.embedding)

    @classmethod
    def load_from_audio(cls, audio: Audio, in_dir: str | Path) -> "SpeakerEmbedding":
        """Loads a SpeakerEmbedding from a file, given the audio object."""
        audio_id = audio.origin.id
        path = Path(in_dir) / f"{audio_id}.npy"
        embedding = np.load(str(path))
        return cls(embedding)
