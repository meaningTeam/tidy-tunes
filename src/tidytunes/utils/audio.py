from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .tensors import collate_tensors

Seconds = float
SEP = "="


@dataclass
class OriginMetadata:
    """Metadata about the original audio source."""

    id: str
    start: float = 0.0
    end: float | None = None


@dataclass
class Segment:
    """Defines a segment of an audio file."""

    start: Seconds
    duration: Seconds
    symbol: any


@dataclass
class Audio:
    """Represents an audio signal with metadata."""

    data: torch.Tensor
    sampling_rate: int
    origin: OriginMetadata | None = None

    @classmethod
    def from_array(
        cls,
        arr: torch.Tensor | np.ndarray,
        sampling_rate: int,
        origin: OriginMetadata | None = None,
    ) -> "Audio":
        """Creates an Audio object from a NumPy array or PyTorch tensor."""
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        if arr.ndim > 1:
            arr = arr.squeeze(0)
            assert arr.ndim == 1
        return cls(arr, sampling_rate, origin)

    @classmethod
    def from_file(cls, path: str | Path, sampling_rate: int | None = None) -> "Audio":
        """
        Loads an audio file and creates an Audio object with metadata.

        If the input file path follows the format like: id=start=duration.*, the
        metadata are initialized accordingly. Otherwise, the whole file name is
        is used as the Audio object id starting at 0.0 seconds.
        """
        pth = Path(path)
        name = pth.stem
        if SEP in name:
            try:
                id, start, end = name.split(SEP)
                start, end = int(start), int(end)
            except:
                name = name.replace(SEP, "_")
                id, start, end = name, 0, None
        else:
            id, start, end = name, 0, None

        audio, sr = torchaudio.load(str(path))
        if audio.shape[0] > 1:
            audio = audio.mean(0)

        audio_obj = cls.from_array(audio, sr, OriginMetadata(id, start, end))
        return audio_obj.resample(sampling_rate) if sampling_rate else audio_obj

    def to_file(
        self, root: str | Path | None = None, path: str | Path | None = None
    ) -> Path:
        """
        Saves the audio data to a file in FLAC format.

        Args:
            root (str | Path, optional): Output directory, used only if the output file
                path is not provided. Output file name is composed from metadata in the
                format like: `root / id=start=duration.flac`
            path: (str | Path, optional): Output file destination, composed from
                metadata if not provided.
        """

        assert (root is None) ^ (
            path is None
        ), "Either root or path must be provided, but not both."

        if root is not None:
            assert (
                self.origin is not None
            ), "Origin metadata is required when using root."
            path = (
                Path(root)
                / f"{self.origin.id}{SEP}{int(self.origin.start)}{SEP}{int(self.origin.end or 0)}.flac"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(path), self.data.unsqueeze(0), self.sampling_rate, format="flac"
        )
        return path

    @property
    def duration(self) -> float:
        """Returns the duration of the audio in seconds."""
        return self.data.shape[-1] / self.sampling_rate

    def as_tensor(self) -> torch.Tensor:
        """Returns the audio data as a PyTorch tensor."""
        return self.data

    def resample(self, sampling_rate: int) -> "Audio":
        """Resamples the audio to a new sampling rate."""
        if sampling_rate == self.sampling_rate:
            return self
        resampled_data = torchaudio.functional.resample(
            self.data.unsqueeze(0), self.sampling_rate, sampling_rate
        )
        return Audio.from_array(
            resampled_data.squeeze(0), sampling_rate, origin=self.origin
        )

    def trim_to_segments(self, segments: list[Segment]) -> list["Audio"]:
        """Trims the audio based on given segments and returns a list of Audio objects."""
        trimmed = []
        for segment in segments:
            start = int(self.sampling_rate * segment.start)
            end = int(self.sampling_rate * (segment.start + segment.duration))

            new_origin = OriginMetadata(
                id=self.origin.id,
                start=self.origin.start + segment.start,
                end=self.origin.start + segment.start + segment.duration,
            )
            trimmed.append(
                Audio.from_array(
                    self.data[start:end], self.sampling_rate, origin=new_origin
                )
            )
        return trimmed

    def play(self):
        """Plays the audio within a Jupyter Notebook."""
        from IPython.display import Audio as _Audio
        from IPython.display import display

        display(_Audio(self.data.numpy(), rate=self.sampling_rate, normalize=False))


def collate_audios(
    audios: list[Audio], sampling_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collates a list of Audio objects into a batched tensor padded with zeros."""
    resampled = [a.resample(sampling_rate).as_tensor() for a in audios]
    return collate_tensors(resampled, 0.0)


def decollate_audios(
    audio: torch.Tensor,
    audio_lens: torch.Tensor,
    sampling_rate: int,
    target_sampling_rate: int | None = None,
    origin_like: list[Audio] | None = None,
) -> list[Audio]:
    """
    Decollates a batched audio tensor into individual Audio objects.

    Args:
        audio (B, T): Batched audio tensor.
        audio_lens: (B,): Lenghts of valid audio samples for each batch item.
        sampling_rate (int): Sampling rate of the audio in the input tensor.
        target_sampling_rate (int, optional): If specified, outputs audios will be
                                              resampled to match this value.
        origin_like (list[Audio], optional): If specified, output Audio objects will
                                             inherit metadata from these Audio objects.

    Returns:
       list[Audio]: List of Audio objects with audio signals in the input audio batch.
    """

    target_sampling_rate = target_sampling_rate or sampling_rate
    decollated = [
        Audio.from_array(a[:l], sampling_rate).resample(target_sampling_rate)
        for a, l in zip(audio, audio_lens)
    ]
    if origin_like is not None:
        assert len(origin_like) == len(decollated)
        for dec, orig in zip(decollated, origin_like):
            dec.origin = orig.origin
    return decollated


def trim_audios(audios: list[Audio], segments: list[list[Segment]]) -> list[Audio]:
    """Trims multiple Audio objects using corresponding Segment lists."""
    assert len(audios) == len(segments)
    return [
        audio for a, segs in zip(audios, segments) for audio in a.trim_to_segments(segs)
    ]
