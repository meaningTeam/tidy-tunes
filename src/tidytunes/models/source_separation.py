import torch
import torch.nn as nn
from torchaudio.transforms import Fade

from tidytunes.utils import masked_mean, masked_std, sequence_mask


class SourceSeparator(nn.Module):
    def __init__(
        self,
        model,
        segment: float = 10.0,
        overlap: float = 0.1,
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate
        self.chunk_len = int(sampling_rate * segment * (1 + overlap))
        self.overlap_frames = int(overlap * sampling_rate)
        self.fade = Fade(
            fade_in_len=0, fade_out_len=self.overlap_frames, fade_shape="linear"
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        """
        Normalizes and processes input audio to separate sources.

        Args:
            x (B, T): Input audio tensor.
            x_lens (B,): Lengths of each sequence in the batch.

        Returns:
            (B, sources, L): Separated audio sources.
        """

        mask = sequence_mask(x_lens)
        mean = masked_mean(x, mask)
        std = masked_std(x, mask, mean=mean)

        x = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        x[~mask] = 0.0

        y = self.get_sources(x)
        y = y * std[:, None, None] + mean[:, None, None]
        mask = mask.unsqueeze(1).repeat(1, 4, 1)
        y[~mask] = 0.0

        # (B, sources, T), sources are: drums, bass, other, vocals
        return y

    @torch.no_grad()
    def get_sources(
        self,
        audio: torch.Tensor,
    ):
        """
        Splits audio into overlapping chunks, processes with the model,
        and applies fade-in/fade-out to smooth transitions.

        Args:
            audio (B, T): Normalized input audio.

        Returns:
            (B, sources, T): Separated sources.
        """

        # The model expects stereo inputs
        audio = audio.unsqueeze(1).expand(-1, 2, -1)
        B, C, L = audio.shape

        if L <= self.chunk_len:
            return self.model(audio).mean(dim=-2)

        output = torch.zeros(B, len(self.model.sources), C, L, device=audio.device)
        start, end = 0, self.chunk_len
        while start < L - self.overlap_frames:
            chunk = audio[:, :, start:end]
            out = self.model(chunk)
            out = self.fade(out)
            output[:, :, :, start:end] += out

            if start == 0:
                self.fade.fade_in_len = self.overlap_frames
                start += self.chunk_len - self.overlap_frames
            else:
                start += self.chunk_len

            end += self.chunk_len
            if end >= L:
                # Disable fade-out for last chunk
                self.fade.fade_out_len = 0

        return output.mean(dim=-2)
