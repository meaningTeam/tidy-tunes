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

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from tidytunes.utils import TraceMixin


class SileroVAD(nn.Module, TraceMixin):
    def __init__(
        self,
        hidden_dim: int = 64,
        features_dim: int = 258,
        num_layers: int = 2,
        audio_padding_size: int = 96,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ["none", "mean", "sum", "max"]
        self.reduction = reduction
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.audio_padding_size = audio_padding_size  # 96 samples ~ 6 ms
        self.sampling_rate = 16000
        self.input_conv = nn.Conv1d(1, features_dim, 256, stride=hidden_dim, bias=False)
        self.mean_conv = nn.Conv1d(1, 1, 7, bias=False)
        self.conv_blocks = nn.Sequential(
            ConvBlock(features_dim, hidden_dim // 4, True, 2),
            ConvBlock(hidden_dim // 4, hidden_dim // 2, True, 2),
            ConvBlock(hidden_dim // 2, hidden_dim // 2, False, 2),
            ConvBlock(hidden_dim // 2, hidden_dim, True, 1),
        )
        self.output_conv = nn.Conv1d(hidden_dim, 1, 1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)

    @classmethod
    def from_files(cls, model_weights_path) -> "SileroVAD":
        vad = cls()
        vad.load_state_dict(
            torch.load(model_weights_path, map_location="cpu", weights_only=True)
        )
        return vad

    @torch.jit.export
    def init_state(
        self, batch: int, device: str = "cpu", dtype: torch.dtype = torch.float
    ) -> list[torch.Tensor]:
        assert dtype == torch.float
        return [
            torch.zeros(
                self.num_layers, batch, self.hidden_dim, device=device, dtype=dtype
            ),
            torch.zeros(
                self.num_layers, batch, self.hidden_dim, device=device, dtype=dtype
            ),
        ]

    def dummy_inputs(
        self,
        batch: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        assert dtype == torch.float
        audio_16khz = torch.randn(batch, 2560, device=device, dtype=dtype)
        state = self.init_state(batch, device, dtype=dtype)
        return (audio_16khz, state)

    def forward(self, audio_chunk_16khz: torch.Tensor, state: list[torch.Tensor]):

        x = audio_chunk_16khz.unsqueeze(1)

        x = F.pad(
            x.float().contiguous(),
            (self.audio_padding_size, self.audio_padding_size),
            mode="reflect",
        )
        x = self.input_conv(x)

        a, b = torch.pow(x, 2).chunk(2, dim=1)
        mag = (a + b).sqrt()
        norm = (mag * (2**20) + 1).log()
        mean = norm.mean(dim=1, keepdim=True)

        # reflact pad 3 frames on both sides, one frame has 4 ms
        left_pad = torch.flip(mean[..., 1:4], dims=[2])
        right_pad = torch.flip(mean[..., -4:-1], dims=[2])
        mean = torch.concat([left_pad, mean, right_pad], dim=2)

        mean = self.mean_conv(mean)
        norm = norm - mean.mean(dim=-1, keepdim=True)

        x = torch.cat([mag, norm], dim=1)
        x = self.conv_blocks(x)

        # b c t -> t b c
        x = x.permute(2, 0, 1)
        x, state = self.lstm(x, state)
        # t b c -> b c t
        x = x.permute(1, 2, 0)

        x = F.relu(x)
        x = self.output_conv(x).squeeze(1)
        y = F.sigmoid(x)

        # reduce probability over each 32 ms chunk of the input
        if self.reduction == "mean":
            y = y.mean(keepdim=False, dim=1)
        elif self.reduction == "sum":
            y = y.sum(keepdim=False, dim=1)
        elif self.reduction == "max":
            y, _ = y.max(keepdim=False, dim=1)

        return y, state


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_residual: bool, stride: int):
        super().__init__()
        self.residual_conv = None
        if use_residual:
            self.residual_conv = nn.Conv1d(in_dim, out_dim, 1)
        self.conv1 = nn.Conv1d(in_dim, in_dim, 5, groups=in_dim, padding=2)
        self.conv2 = nn.Conv1d(in_dim, out_dim, 1)
        self.conv3 = nn.Conv1d(out_dim, out_dim, 1, stride=stride)

    def forward(self, x: torch.Tensor):
        x2 = F.relu(self.conv1(x))
        if self.residual_conv is not None:
            x = self.residual_conv(x)
        x = F.relu(self.conv2(x2) + x)
        x = F.relu(self.conv3(x))
        return x
