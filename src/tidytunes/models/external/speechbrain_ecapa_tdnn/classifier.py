"""
Classifier to be used on top of ECAPA TDNN:
https://github.com/speechbrain/speechbrain/blob/f07cfc76bd4b864c598a9ed5948caa3fe3176516/speechbrain/lobes/models/Xvector.py#L118
https://github.com/speechbrain/speechbrain/blob/f07cfc76bd4b864c598a9ed5948caa3fe3176516/speechbrain/nnet/containers.py#L20

The original code is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

from tidytunes.models.external.speechbrain_ecapa_tdnn.layers import (
    BatchNorm1d as _BatchNorm1d,
)
from tidytunes.models.external.speechbrain_ecapa_tdnn.layers import (
    Linear,
    Sequential,
    Softmax,
)


class BatchNorm1d(_BatchNorm1d):
    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
    ):
        super().__init__(
            input_shape=input_shape,
            input_size=input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            combine_batch_time=combine_batch_time,
            skip_transpose=False,
        )


class Classifier(Sequential):

    def __init__(
        self,
        input_shape=[None, None, 256],
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=107,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(Sequential, layer_name="DNN")

        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(Sequential, layer_name=block_name)
            self.DNN[block_name].append(
                Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(BatchNorm1d, layer_name="norm")

        # Final Softmax classifier
        self.append(Linear, n_neurons=out_neurons, layer_name="out")
        self.append(Softmax(apply_log=False), layer_name="softmax")
