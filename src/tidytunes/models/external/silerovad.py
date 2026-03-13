"""
This file contains code that follows the model design of SileroVAD,
available at https://github.com/snakers4/silero-vad.

SileroVAD is released under the MIT license, and while this implementation
is not directly copied from the repository, it is inspired by the original
work.

MIT License for SileroVAD:

Copyright (c) 2021 Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidytunes.utils import TraceMixin


class STFT(nn.Module):
    """Magnitude STFT via a windowed DFT basis stored as a conv1d filter."""

    def __init__(self, n_fft: int = 256, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_right = n_fft // 4

        num_bins = n_fft // 2 + 1
        window = torch.hann_window(n_fft, periodic=True)
        n = torch.arange(n_fft, dtype=torch.float32)

        real_basis = torch.zeros(num_bins, n_fft)
        imag_basis = torch.zeros(num_bins, n_fft)
        for k in range(num_bins):
            phase = 2.0 * math.pi * k * n / n_fft
            real_basis[k] = torch.cos(phase) * window
            imag_basis[k] = -torch.sin(phase) * window

        forward_basis = torch.cat([real_basis, imag_basis], dim=0).unsqueeze(1)
        self.register_buffer("forward_basis_buffer", forward_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [0, self.pad_right], mode="reflect")
        x = x.unsqueeze(1)
        ft = F.conv1d(x, self.forward_basis_buffer, stride=self.hop_length)
        cutoff = self.n_fft // 2 + 1
        real = ft[:, :cutoff, :].float()
        imag = ft[:, cutoff:, :].float()
        return torch.sqrt((real.pow(2) + imag.pow(2)).clamp_min(1e-16))


class SileroVADv6(nn.Module, TraceMixin):

    def __init__(
        self,
        context_size: int = 64,
        chunk_size: int = 512,
        n_fft: int = 256,
        hop_length: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.context_size = context_size
        self.hidden_dim = hidden_dim

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(129, 128, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=False)
        self.decoder_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @property
    def frame_shift(self):
        return self.chunk_size / self.sampling_rate

    @property
    def sampling_rate(self):
        return 16000

    @torch.jit.export
    def init_state(self, batch: int = 1, device: str = "cpu") -> list[torch.Tensor]:
        return [
            torch.zeros(batch, self.context_size, device=device),
            torch.zeros(1, batch, self.hidden_dim, device=device, dtype=torch.float),
            torch.zeros(1, batch, self.hidden_dim, device=device, dtype=torch.float),
        ]

    def dummy_inputs(
        self,
        batch: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a ``(audio_chunk, state)`` pair suitable for :meth:`forward`."""
        audio_16khz = torch.randn(
            batch,
            self.chunk_size,
            device=device,
            dtype=dtype,
        )
        state = self.init_state(batch, device=device)
        return (audio_16khz, state)

    def _encode_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """Run STFT + conv encoder. Input ``[..., samples]``, output ``[..., 128]``."""
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.stft(x)
        for conv in self.encoder:
            x = F.relu(conv(x))
        return x.squeeze(-1).reshape(*shape, self.hidden_dim)

    def forward(
        self,
        audio_chunk_16khz: torch.Tensor,
        state: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:

        x = torch.cat([state[0], audio_chunk_16khz], dim=1)
        new_context = x[:, -self.context_size :]

        x = self._encode_chunks(x)
        x = x.unsqueeze(0)  # [1, B, H] for LSTM

        _, (h, c) = self.rnn(x, (state[1], state[2]))

        x = h.squeeze(0).unsqueeze(-1).float()
        x = self.decoder_head(x)
        out = x.squeeze(1).mean(dim=-1)

        return out, [new_context, h, c]

    @classmethod
    def from_files(
        cls,
        model_weights_path: str,
    ) -> "SileroVADv6":
        model = cls()
        sd = torch.load(model_weights_path, map_location="cpu", weights_only=True)
        sd = {
            {
                "rnn.weight_ih": "rnn.weight_ih_l0",
                "rnn.weight_hh": "rnn.weight_hh_l0",
                "rnn.bias_ih": "rnn.bias_ih_l0",
                "rnn.bias_hh": "rnn.bias_hh_l0",
            }.get(k, k): v
            for k, v in sd.items()
        }
        model.load_state_dict(sd)
        return model
