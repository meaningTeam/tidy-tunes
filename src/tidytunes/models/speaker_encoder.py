import torch
import torch.nn.functional as F

from tidytunes.models.external.resnet_speaker_encoder import ResNetSpeakerEncoder


class SpeakerEncoder(torch.nn.Module):

    def __init__(
        self,
        model: ResNetSpeakerEncoder,
        window_size: int,
        hop_length: int,
        sampling_rate: int,
    ):
        """
        Speaker embedding encoder using a ResNet-based x-vector extractor.

        Args:
            model (ResNetSpeakerEncoder): Pre-trained speaker encoder model.
            window_size (int): Size of the sliding window in model frames.
            hop_length (int): Hop length (samples per frame).
            sampling_rate (int): Sampling rate of the input audio.
        """
        super().__init__()
        self.model = model
        self.hop_length = hop_length
        self.window_size = window_size
        self.sampling_rate = sampling_rate

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
        Computes speaker embeddings for input audio sequences.

        Args:
            x (torch.Tensor): Input batch of audio waveforms (B, T).
            x_lens (torch.Tensor): Lengths of valid audio in each batch (B,).

        Returns:
            List[torch.Tensor]: Speaker embeddings for each input in the batch.
        """
        chunks_flat, lens = [], []

        for a, al in zip(x, x_lens):
            chunks = self.split_to_chunks(a[:al])
            chunks_flat.append(chunks)
            lens.append(chunks.shape[0])

        chunks_flat = torch.cat(chunks_flat, dim=0)
        embeddings_flat = self.model(chunks_flat)
        embeddings = torch.split(embeddings_flat, lens)

        return embeddings

    def split_to_chunks(self, x: torch.Tensor):
        pad_size = (self.window_size - 1) * self.hop_length
        x = F.pad(
            x.unsqueeze(0), (pad_size, pad_size + self.hop_length - 1), mode="reflect"
        ).squeeze(0)
        frames_flat = x.unfold(0, self.window_size * self.hop_length, self.hop_length)
        return frames_flat
