"""
Embedding extraction backbone.
This file contains slightly modified code code available at:
https://github.com/speechbrain/speechbrain/blob/f07cfc76bd4b864c598a9ed5948caa3fe3176516/speechbrain/nnet/CNN.py#L312
https://github.com/speechbrain/speechbrain/blob/f07cfc76bd4b864c598a9ed5948caa3fe3176516/speechbrain/nnet/normalization.py#L13

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

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


class Sequential(torch.nn.ModuleDict):
    """A sequence of modules with potentially inferring shape on construction.

    If layers are passed with names, these can be referenced with dot notation.

    Arguments
    ---------
    *layers : tuple
        Layers to be applied in sequence.
    input_shape : iterable
        A list or tuple of ints or None, representing the expected shape of an
        input tensor. None represents a variable-length dimension. If no
        ``input_shape`` is passed, no shape inference will be performed.
    **named_layers : dict
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer. If a tuple is returned,
        only the shape of the first element is used to determine input
        shape of the next layer (e.g. RNN returns output, hidden).

    Example
    -------
    >>> inputs = torch.rand(10, 40, 50)
    >>> model = Sequential(input_shape=inputs.shape)
    >>> model.append(Linear, n_neurons=100, layer_name="layer1")
    >>> model.append(Linear, n_neurons=200, layer_name="layer2")
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 200])
    >>> outputs = model.layer1(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 100])
    """

    def __init__(self, *layers, input_shape=None, **named_layers):
        super().__init__()

        # Make sure either layers or input_shape is passed
        if not layers and input_shape is None and not named_layers:
            raise ValueError("Must pass either layers or input shape")

        # Keep track of what layers need "lengths" passed
        self.length_layers = []

        # Replace None dimensions with arbitrary value
        self.input_shape = input_shape
        if input_shape and None in input_shape:
            self.input_shape = list(input_shape)
            for i, dim in enumerate(self.input_shape):
                # To reduce size of dummy tensors, use 1 for batch dim
                if i == 0 and dim is None:
                    dim = 1

                # Use 64 as nice round arbitrary value, big enough that
                # halving this dimension a few times doesn't reach 1
                self.input_shape[i] = dim or 256

        # Append non-named layers
        for layer in layers:
            self.append(layer)

        # Append named layers
        for name, layer in named_layers.items():
            self.append(layer, layer_name=name)

    def append(self, layer, *args, layer_name=None, **kwargs):
        """Add a layer to the list of layers, inferring shape if necessary.

        Arguments
        ---------
        layer : A torch.nn.Module class or object
            If the layer is a class, it should accept an argument called
            ``input_shape`` which will be inferred and passed. If the layer
            is a module object, it is added as-is.
        *args : tuple
            These are passed to the layer if it is constructed.
        layer_name : str
            The name of the layer, for reference. If the name is in use,
            ``_{count}`` will be appended.
        **kwargs : dict
            These are passed to the layer if it is constructed.
        """

        # Compute layer_name
        if layer_name is None:
            layer_name = str(len(self))
        elif layer_name in self:
            index = 0
            while f"{layer_name}_{index}" in self:
                index += 1
            layer_name = f"{layer_name}_{index}"

        # Check if it needs to be constructed with input shape
        if self.input_shape:
            argspec = inspect.getfullargspec(layer)
            if "input_shape" in argspec.args + argspec.kwonlyargs:
                input_shape = self.get_output_shape()
                layer = layer(*args, input_shape=input_shape, **kwargs)

        # Finally, append the layer.
        try:
            self.add_module(layer_name, layer)
        except TypeError:
            raise ValueError(
                "Must pass `input_shape` at initialization and use "
                "modules that take `input_shape` to infer shape when "
                "using `append()`."
            )

    def get_output_shape(self):
        """Returns expected shape of the output.

        Computed by passing dummy input constructed with the
        ``self.input_shape`` attribute.

        Returns
        -------
        Expected shape of the output after all layers applied.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            dummy_output = self(dummy_input)
        return dummy_output.shape

    def forward(self, x):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.

        Returns
        -------
        x : torch.Tensor
            Output after all layers are applied.
        """
        for layer in self.values():
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=True,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got " + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError("conv1d expects 2d, 3d inputs. Got " + str(len(shape)))

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    skip_transpose : bool
        Whether to skip the transposition.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=True,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.

        Returns
        -------
        x_n : torch.Tensor
            The normalized outputs.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[3], shape_or[2])

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape : tuple
        It is the shape of the input tensor.
    input_size : int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    max_norm : float
        weight max-norm.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        max_norm=None,
        combine_dims=False,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.

        Returns
        -------
        wx : torch.Tensor
            The linearly transformed outputs.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        if self.max_norm is not None:
            self.w.weight.data = torch.renorm(
                self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.w(x)

        return wx


class Softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        If the dimension where softmax is applied.
    reshape: bool
        whether to apply reshaping (true by default)
    dtype: torch.dtype
        dtype of the output tensor

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, apply_log=False, dim=-1, reshape=True, dtype=torch.float32):
        super().__init__()

        if apply_log:
            self.act = F.log_softmax
        else:
            self.act = F.softmax

        self.dim = dim
        self.reshape = reshape
        self.dtype = dtype

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x_act : torch.Tensor
            The softmax outputs.
        """
        # Reshaping the tensors
        dims = x.shape

        if self.reshape:
            if len(dims) == 3:
                x = x.reshape(dims[0] * dims[1], dims[2])

            if len(dims) == 4:
                x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x, dim=self.dim, dtype=self.dtype)

        # Retrieving the original shape format
        if self.reshape:
            if len(dims) == 3:
                x_act = x_act.reshape(dims[0], dims[1], dims[2])

            if len(dims) == 4:
                x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act
