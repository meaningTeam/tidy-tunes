"""
Classes related to mean-var normalization.
This file contains slightly modified code code available at:
https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/features.py

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

from tidytunes.utils import sequence_mask


class InputNormalization(torch.nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.
    norm_type : str
         It defines how the statistics are computed ('sentence' computes them
         at sentence level, 'batch' at batch level, 'speaker' at speaker
         level, while global computes a single normalization vector for all
         the sentences in the dataset). Speaker and global statistics are
         computed with a moving average approach.
    avg_factor : float
         It can be used to manually set the weighting factor between
         current statistics and accumulated ones.
    requires_grad : bool
        Whether this module should be updated using the gradient during training.
    update_until_epoch : int
        The epoch after which updates to the norm stats should stop.

    Example
    -------
    >>> import torch
    >>> norm = InputNormalization()
    >>> inputs = torch.randn([10, 101, 20])
    >>> inp_len = torch.ones([10])
    >>> features = norm(inputs, inp_len)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, torch.Tensor]
    spk_dict_std: Dict[int, torch.Tensor]
    spk_dict_count: Dict[int, int]

    def __init__(
        self,
        mean_norm=True,
        std_norm=False,
        avg_factor=None,
        requires_grad=False,
        update_until_epoch=3,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.glob_mean = torch.tensor([0])
        self.glob_std = torch.tensor([0])
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths, spk_ids=torch.tensor([]), epoch=0):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.
        lengths : torch.Tensor
            A batch of tensors containing the relative length of each
            sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
            computing stats on zero-padded steps.
        spk_ids : torch.Tensor containing the ids of each speaker (e.g, [0 10 6]).
            It is used to perform per-speaker normalization when
            norm_type='speaker'.
        epoch : int
            The epoch count.

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """

        real_lengths = torch.round(lengths * x.shape[1]).long()
        current_means, current_std = self._compute_current_stats(x, real_lengths)
        out = (x - current_means) / current_std

        return out

    def _compute_current_stats(self, x, x_lens):

        mask = sequence_mask(x_lens, max_length=x.shape[1])

        # Compute current mean
        if self.mean_norm:
            current_mean = x.sum(dim=1, keepdim=True) / (mask).sum(
                dim=1, keepdim=True
            ).unsqueeze(-1)
            current_mean = current_mean.detach()
        else:
            current_mean = torch.zeros_like(x_lens).float()
            current_mean = current_mean.view(-1, 1, 1)

        # Compute current std
        if self.std_norm:
            raise NotImplementedError()
        else:
            current_std = torch.ones_like(x_lens).float()
            current_std = current_std.view(-1, 1, 1)

        # Improving numerical stability of std
        current_std = torch.clamp(current_std, min=self.eps)

        return current_mean, current_std

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics."""
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.

        Returns
        -------
        state : dict
        """
        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  # .to(self.device_inp)
            self.glob_std = state["glob_std"]  # .to(self.device_inp)

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]  # .to(
            #    self.device_inp
            # )

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk]  # .to(
            #    self.device_inp
            # )

        self.spk_dict_count = state["spk_dict_count"]

        return state

    def to(self, device):
        """Puts the needed tensors in the right device."""
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)
        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_std[spk] = self.spk_dict_std[spk].to(device)
        return self

    def _save(self, path):
        """Save statistic dictionary.

        Arguments
        ---------
        path : str
            A path where to save the dictionary.
        """
        stats = self._statistics_dict()
        torch.save(stats, path)

    def load(self, path, map_location: str = "cpu"):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        end_of_epoch : bool
            Whether this is the end of an epoch.
            Here for compatibility, but not used.
        """
        stats = torch.load(path, map_location=map_location, weights_only=True)
        self._load_statistics_dict(stats)
