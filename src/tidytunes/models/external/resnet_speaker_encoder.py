"""
This file contains code that is based on the Speaker Encoder model from
Coqui AI, available at:
https://github.com/coqui-ai/TTS/wiki/Speaker-Encoder

The Coqui AI Speaker Encoder is released under the Mozilla Public License
Version 2.0 (MPL-2.0). Parts of this work are also inspired by the
VoxCeleb Trainer, originally available at:
https://github.com/clovaai/voxceleb_trainer, which is released under the
MIT License.

MPL-2.0 License for Coqui AI Speaker Encoder:
---------------------------------------------
Copyright (c) 2021 Coqui AI

MIT License for VoxCeleb Trainer:
---------------------------------
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from tidytunes.utils import TraceMixin


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter",
            torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module, TraceMixin):
    """
    Implementation of the model H/ASP without batch normalization in speaker embedding.
    This model was proposed in: https://arxiv.org/abs/2009.14153
    """

    def __init__(
        self,
        input_dim: int = 64,
        proj_dim: int = 512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input: bool = True,
        l2_norm: bool = True,
        num_input_frames: int = 64,
        hop_length: int = 160,
        preemphasis: float = 0.97,
        sample_rate: int = 16000,
        fft_size: int = 512,
        win_length: int = 400,
        num_mels: int = 64,
    ):
        super().__init__()

        self.num_input_frames = num_input_frames
        self.l2_norm = l2_norm
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.proj_dim = proj_dim
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(
            SEBasicBlock, num_filters[1], layers[1], stride=(2, 2)
        )
        self.layer3 = self.create_layer(
            SEBasicBlock, num_filters[2], layers[2], stride=(2, 2)
        )
        self.layer4 = self.create_layer(
            SEBasicBlock, num_filters[3], layers[3], stride=(2, 2)
        )

        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.torch_spec = torch.nn.Sequential(
            PreEmphasis(preemphasis),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=fft_size,
                win_length=win_length,
                hop_length=self.hop_length,
                window_fn=torch.hamming_window,
                n_mels=num_mels,
            ),
        )

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the model (chunked input only).

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """

        x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if self.l2_norm:
            x = torch.nn.functional.normalize(x, p=2.0, dim=1)
        return x

    def dummy_inputs(
        self,
        batch: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        return (
            torch.randn(
                batch,
                self.hop_length * self.num_input_frames,
                device=device,
                dtype=dtype,
            ),
        )

    @classmethod
    def from_files(cls, model_weights_path) -> "ResNetSpeakerEncoder":
        model = cls()
        sd = torch.load(model_weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd["model"])
        return model
