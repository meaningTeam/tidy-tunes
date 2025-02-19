import torch

from tidytunes.utils.trace import TraceMixin


class RollOff(torch.nn.Module, TraceMixin):
    def __init__(self, sampling_rate: int, roll_percent: float):
        """
        Computes the spectral roll-off frequency, which is the frequency below
        which a given percentage of the total spectral energy is contained.

        Args:
            sampling_rate (int): Sampling rate of the audio signal.
            roll_percent (float): Percentage of spectral energy to consider.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.roll_percent = roll_percent

    def forward(self, audio: torch.Tensor):
        """
        Computes the roll-off frequency for a given audio signal.

        Args:
            audio (torch.Tensor): Input audio waveform (1, T).

        Returns:
            torch.Tensor: Roll-off frequency (Hz).
        """

        audio = audio.squeeze(0)
        assert audio.ndim == 1, "Expected single channel waveform of shape (1, T)"

        fft = torch.fft.fft(audio)
        magnitude = torch.abs(fft)
        freq = torch.fft.fftfreq(magnitude.shape[0], 1 / self.sampling_rate)

        positive_freqs = freq[: freq.shape[0] // 2]
        positive_magnitude = magnitude[: magnitude.shape[0] // 2]

        total_energy = torch.sum(positive_magnitude, dim=-1)
        cumulative_energy = torch.cumsum(positive_magnitude, dim=-1)
        normalized_energy = cumulative_energy / total_energy

        rolloff_index = torch.searchsorted(normalized_energy, self.roll_percent)
        rolloff_freq = positive_freqs[rolloff_index]

        return rolloff_freq.unsqueeze(0)

    def dummy_inputs(
        self,
        batch: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        return torch.randn(1, self.sampling_rate * 10, device=device, dtype=dtype)
