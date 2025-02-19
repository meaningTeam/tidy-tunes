"""
This file is based on code from the attenuate 0.1.1 package, 
available at https://pypi.org/project/attenuate/.

The concepts implemented in this code are described in the paper:
"Real-time Speech Enhancement on Raw Signals with Deep State-space
Modeling" (https://arxiv.org/pdf/2409.03377).

Apache License, Version 2.0 for aTENNuate:
------------------------------------------
Copyright (c) BrainChip

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import numpy as np
import torch
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class AttenuateDenoiser(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=[16, 32, 64, 96, 128, 256],
        num_coeffs=16,
        repeat=16,
        resample_factors=[4, 4, 2, 2, 2, 2],
        pre_conv=True,
        sampling_rate=16000,
    ):
        super().__init__()

        depth = len(channels)
        self.depth = depth
        self.channels = [in_channels] + channels
        self.num_coeffs = num_coeffs
        self.repeat = repeat
        self.pre_conv = pre_conv
        self.sampling_rate = sampling_rate

        self.down_ssms = nn.ModuleList(
            [
                self.ssm_pool(c_in, c_out, r, downsample=True)
                for (c_in, c_out, r) in zip(
                    self.channels[:-1], self.channels[1:], resample_factors
                )
            ]
        )
        self.up_ssms = nn.ModuleList(
            [
                self.ssm_pool(c_in, c_out, r, downsample=False)
                for (c_in, c_out, r) in zip(
                    self.channels[1:], self.channels[:-1], resample_factors
                )
            ]
        )
        self.hid_ssms = nn.Sequential(
            self.ssm_block(self.channels[-1], True),
            self.ssm_block(self.channels[-1], True),
        )
        self.last_ssms = nn.Sequential(
            self.ssm_block(self.channels[0], True),
            self.ssm_block(self.channels[0], False),
        )

    @torch.no_grad()
    def forward(self, audio, mask=None):
        assert audio.ndim == 2, f"audio input should be shaped (batch, samples)"
        audio = audio[:, None, :]  # unsqueeze channel dim

        padding = 256 - audio.shape[-1] % 256
        noisy = F.pad(audio, (0, padding))
        denoised = self.forward_chunk(noisy)
        denoised = denoised.squeeze(1)[..., :-padding]
        if mask is not None:
            denoised[~mask] = 0.0
        return denoised

    def forward_chunk(self, input):
        x, skips = input, []

        # encoder
        for ssm in self.down_ssms:
            skips.append(x)
            x = ssm(x)

        # neck
        x = self.hid_ssms(x)

        # decoder
        for ssm, skip in zip(self.up_ssms[::-1], skips[::-1]):
            x = ssm[0](x)
            x = x + skip
            x = ssm[1](x)

        return self.last_ssms(x)

    def ssm_pool(self, in_channels, out_channels, resample_factor, downsample=True):
        if downsample:
            return nn.Sequential(
                self.ssm_block(in_channels, use_activation=True),
                EinMix(
                    "b c (t r) -> b d t",
                    weight_shape="c d r",
                    c=in_channels,
                    d=out_channels,
                    r=resample_factor,
                ),
            )
        else:
            return nn.Sequential(
                EinMix(
                    "b c t -> b d (t r)",
                    weight_shape="c d r",
                    c=in_channels,
                    d=out_channels,
                    r=resample_factor,
                ),
                self.ssm_block(out_channels, use_activation=True),
            )

    def ssm_block(self, channels, use_activation=False):
        block = nn.Sequential()
        if channels > 1 and self.pre_conv:
            block.append(nn.Conv1d(channels, channels, 3, 1, 1, groups=channels))
        block.append(SSMLayer(self.num_coeffs, channels, channels, self.repeat))
        if use_activation:
            if channels > 1:
                block.append(LayerNormFeature(channels))
            block.append(nn.SiLU())

        return block

    @classmethod
    def from_files(cls, model_weights_path) -> "AttenuateDenoiser":
        model = cls()
        model.load_state_dict(
            torch.load(model_weights_path, map_location="cpu", weights_only=True)
        )
        return model


class SSMLayer(nn.Module):
    def __init__(
        self, num_coeffs: int, in_channels: int, out_channels: int, repeat: int
    ):
        from torch.backends import opt_einsum

        assert opt_einsum.is_available()
        opt_einsum.strategy = "optimal"

        super().__init__()

        init_parameter = lambda mat: Parameter(torch.tensor(mat, dtype=torch.float))
        normal_parameter = lambda fan_in, shape: Parameter(
            torch.randn(*shape) * math.sqrt(2 / fan_in)
        )

        A_real, A_imag = 0.5 * np.ones(num_coeffs), math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1)  # inv softplus
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)

        A = np.tile(A, (repeat, 1))
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)

        self.log_dt, self.A, self.B = (
            init_parameter(log_dt),
            init_parameter(A),
            init_parameter(B),
        )
        self.C = normal_parameter(
            num_coeffs * repeat, (out_channels, num_coeffs * repeat)
        )

    def forward(self, input):
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)


class LayerNormFeature(nn.Module):
    """Apply LayerNorm to the channel dimension"""

    def __init__(self, features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, input):
        return self.layer_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)


@torch.compiler.disable
def fft_conv(equation, input, kernel, *args):
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]

    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)

    return output[..., :n]


def ssm_basis_kernels(A, B, log_dt, length):
    log_A_real, A_imag = A.T  # (2, num_coeffs)
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()

    dtA_real, dtA_imag = -dt * F.softplus(log_A_real), dt * A_imag
    return (dtA_real[:, None] * lrange).exp() * torch.cos(
        dtA_imag[:, None] * lrange
    ), B * dt[:, None]


def opt_ssm_forward(input, K, B_hat, C):
    """SSM ops with einsum contractions"""
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape

    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum("dn,nc,nl->dcl", C, B_hat, K)
            return fft_conv("bcl,dcl->bdl", input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum("bcl,nc->bnl", input, B_hat)
            x = fft_conv("bnl,nl->bnl", x, K)
            return torch.einsum("bnl,dn->bdl", x, C)

    return fft_conv("bcl,nl,nc,dn->bdl", input, K, B_hat, C)
