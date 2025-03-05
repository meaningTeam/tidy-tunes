import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Fade

from tidytunes.utils import masked_mean, masked_std, sequence_mask


class SourceSeparator(nn.Module):
    def __init__(
        self,
        model,
        frame_shift: float = 0.16,
        segment_frames: int = 63,
        overlap_frames: int = 5,
        sampling_rate: int = 44100,
        max_music_energy: float = 0.01,
        min_speech_energy: float = 0.99,
        window_frames: int = 2,
        minimal_energy: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.frame_shift = frame_shift
        self.sampling_rate = sampling_rate
        self.frame_samples = int(frame_shift * sampling_rate)
        self.segment_samples = self.frame_samples * segment_frames
        self.overlap_samples = self.frame_samples * overlap_frames
        self.fade_in = Fade(fade_in_len=self.overlap_samples, fade_out_len=0)
        self.fade_out = Fade(fade_in_len=0, fade_out_len=self.overlap_samples)
        self.max_music_energy = max_music_energy
        self.min_speech_energy = min_speech_energy
        self.window_samples = self.frame_samples * window_frames
        self.minimal_energy = minimal_energy

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ):
        """
        Normalizes and processes input audio to separate sources and calculates
        the energy of vocals and the rest of sources within a sliding window to
        decide if there is a background music together with speech or not.

        Args:
            audio (B, T): Input audio tensor.
            audio_lens (B,): Lengths of each sequence in the batch.

        Returns:
            (B, L): Mask indicating frames without music.
        """

        B, T = audio.shape

        # Pad to nearest multiple of segment and add overlap
        padded_lens = (
            (T + self.segment_samples - 1) // self.segment_samples
        ) * self.segment_samples
        pad_size = padded_lens - T + self.overlap_samples

        audio = F.pad(audio, (0, pad_size))

        mask = sequence_mask(audio_lens, max_length=audio.shape[-1])
        mean = masked_mean(audio, mask)
        std = masked_std(audio, mask, mean=mean)

        x = (audio - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        x[~mask] = 0.0

        audio_buffer = torch.zeros(x.shape[0], self.overlap_samples, device=x.device)
        window_buffer = None
        output = []

        for i in range((x.shape[-1] - self.overlap_samples) // self.segment_samples):

            s = i * self.segment_samples
            e = (i + 1) * self.segment_samples + self.overlap_samples
            segment = x[..., s:e]

            assert segment.shape[-1] == self.segment_samples + self.overlap_samples

            if i > 0:
                segment = self.fade_in(segment)
            segment[..., : self.overlap_samples] += audio_buffer
            segment = self.fade_out(segment)
            audio_buffer = segment[..., -self.overlap_samples :]
            segment = segment[..., : -self.overlap_samples]

            y = self.forward_segment(segment)
            y = y * std[:, None, None] + mean[:, None, None]

            if window_buffer is not None:
                y = torch.cat([window_buffer, y], dim=-1)
            window_buffer = y[..., -self.frame_samples :]

            frames = y.unfold(-1, self.window_samples, self.frame_samples)

            energy = (frames**2).mean(dim=-1)  # b c t w -> b c t

            # Merge all non-vocal channels into a single one
            energy = torch.cat(
                (energy[:, :3].sum(dim=-2, keepdim=True), energy[:, 3:]), dim=-2
            )
            energy_total = energy.sum(dim=-2, keepdim=True)
            rel_energy = energy / energy_total

            abs_silence = energy_total.squeeze(1) < self.minimal_energy
            no_music = abs_silence | (rel_energy[:, 0] <= self.max_music_energy)
            is_speech = abs_silence | (rel_energy[:, 1] >= self.min_speech_energy)
            output.append(no_music & is_speech)

        output = torch.cat(output, dim=-1)

        # Trim output to remove segment padding and mask invalid positions
        n_frames = (
            audio_lens + 2 * self.frame_samples - 1 - self.window_samples
        ) // self.frame_samples
        output = output[..., : n_frames.max()]
        mask = sequence_mask(n_frames)
        output[~mask] = False

        return output

    @torch.no_grad()
    def forward_segment(
        self,
        x: torch.Tensor,
    ):
        # The model expects stereo inputs
        x = x.unsqueeze(1).expand(-1, 2, -1)
        B, C, L = x.shape
        assert L == self.segment_samples
        x = self.model(x).mean(dim=-2)
        return x
