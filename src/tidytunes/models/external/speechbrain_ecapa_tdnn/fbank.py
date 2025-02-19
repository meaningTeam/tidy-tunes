"""
Classes related to low-level feature extraction.
This file contains slightly modified code code available at:
https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/features.py
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

import logging
import math

import torch


def spectral_magnitude(stft, power: int = 1, log: bool = False, eps: float = 1e-14):
    """Returns the magnitude of a complex spectrogram.

    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.
    power : int
        What power to use in computing the magnitude.
        Use power=1 for the power spectrogram.
        Use power=0.5 for the magnitude spectrogram.
    log : bool
        Whether to apply log to the spectral features.
    eps : float
        A small value to prevent square root of zero.

    Returns
    -------
    spectr : torch.Tensor

    Example
    -------
    >>> a = torch.Tensor([[3, 4]])
    >>> spectral_magnitude(a, power=0.5)
    tensor([5.])
    """
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr


class STFT(torch.nn.Module):
    """computes the Short-Term Fourier Transform (STFT).

    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.

    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    torch.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(round((self.sample_rate / 1000.0) * self.win_length))
        self.hop_length = int(round((self.sample_rate / 1000.0) * self.hop_length))

        self.window = window_fn(self.win_length)

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : torch.Tensor
            A batch of audio signals to transform.

        Returns
        -------
        stft : torch.Tensor
        """
        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=True,
        )

        stft = torch.view_as_real(stft)

        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)

        return stft


class Filterbank(torch.nn.Module):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Minimum negative cut-off in decibels.
    param_change_factor : bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor : float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.

    Example
    -------
    >>> import torch
    >>> compute_fbanks = Filterbank()
    >>> inputs = torch.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logging.error(err_msg, exc_info=True)

        # Filter definition
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
            self.f_central = torch.nn.Parameter(
                self.f_central / (self.sample_rate * self.param_change_factor)
            )
            self.band = torch.nn.Parameter(
                self.band / (self.sample_rate * self.param_change_factor)
            )

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # Replicating for all the filters
        self.all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        spectrogram : torch.Tensor
            A batch of spectrogram tensors.

        Returns
        -------
        fbanks : torch.Tensor
        """
        # Computing central frequency and bandwidth of each filter
        f_central_mat = self.f_central.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )
        band_mat = self.band.repeat(self.all_freqs_mat.shape[1], 1).transpose(0, 1)

        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate * self.param_change_factor * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate * self.param_change_factor * self.param_change_factor
            )

        # Regularization with random changes of filter central frequency and band
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(
            spectrogram.device
        )

        sp_shape = spectrogram.shape

        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.permute(0, 3, 1, 2)
            spectrogram = spectrogram.reshape(
                sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]
            )

        # FBANK computation
        fbanks = torch.matmul(spectrogram, fbank_matrix)
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)

        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(sp_shape[0], sp_shape[3], fb_shape[1], fb_shape[2])
            fbanks = fbanks.permute(0, 2, 3, 1)

        return fbanks

    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        hz : float
            The frequency point in Hz.

        Returns
        -------
        The mel-frequency value
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        mel : float
            The frequency point in the mel-scale.

        Returns
        -------
        The hz-frequency value
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        # Computing the slops of the filters
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0

        # Adding zeros for negative values
        zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(zero, torch.min(left_side, right_side)).transpose(0, 1)

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.
        smooth_factor: torch.Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central_mat : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band_mat : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : torch.Tensor
            A batch of linear FBANK tensors.

        Returns
        -------
        x_db : torch.Tensor
        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max. It is the max over time and frequency,
        # Hence, of a whole sequence (sequence-dependent)
        new_x_db_max = x_db.amax(dim=(-2, -1)) - self.top_db

        # Clipping to dB max. The view is necessary as only a scalar is obtained
        # per sequence.
        x_db = torch.max(x_db, new_x_db_max.view(x_db.shape[0], 1, 1))

        return x_db


class Fbank(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: False)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: False)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 160000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 40)
        Number of Mel filters.
    filter_shape : str (default: triangular)
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor : float (default: 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor : float (default: 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default: 5)
        Number of frames of left context to add.
    right_frames : int (default: 5)
        Number of frames of right context to add.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=60,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )

    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals to transform to features.

        Returns
        -------
        fbanks : torch.Tensor
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)
        return fbanks
