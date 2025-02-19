import torch
import torch.nn as nn

from tidytunes.models.external import SileroVAD


class VoiceActivityDetector(nn.Module):
    def __init__(
        self,
        model: SileroVAD,
        frame_shift: float = 0.16,
        min_silence_chunks: int = 4,
        start_threshold: float = 0.9,
        end_threshold: float = 0.9,
    ):
        """
        Voice Activity Detector using SileroVAD.

        Args:
            vad (SileroVAD): Pre-trained SileroVAD model.
            frame_shift (float): Frame shift duration in seconds.
            min_silence_chunks (int): Minimum number of consecutive silence frames to trigger an end event.
            start_threshold (float): Probability threshold to start speech detection.
            end_threshold (float): Probability threshold to stop speech detection.
        """
        super().__init__()
        assert (
            start_threshold >= end_threshold
        ), "start_threshold must be >= end_threshold"

        self.model = model
        self.frame_shift = frame_shift
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.n_samples = int(frame_shift * model.sampling_rate)
        self.min_silence_samples = min_silence_chunks * self.n_samples

    @property
    def sampling_rate(self):
        return self.model.sampling_rate

    @torch.no_grad()
    def forward(self, audio_16khz):
        """
        Processes an audio signal to detect voice activity.

        Args:
            audio_16khz (B, T): Input audio waveform (assumed to be sampled at 16kHz).

        Returns:
            Binary mask (B, L) indicating speech presence.
        """
        audio_16khz = torch.atleast_2d(audio_16khz)
        self.initialize(len(audio_16khz), str(audio_16khz.device))

        max_length = (audio_16khz.shape[-1] // self.n_samples) * self.n_samples
        audio_chunks = torch.split(audio_16khz[:, :max_length], self.n_samples, dim=-1)
        outs = [self.forward_chunk(chunk) for chunk in audio_chunks]

        is_speech = torch.cat([o[0].unsqueeze(1) for o in outs], dim=-1)
        return is_speech

    @torch.no_grad()
    def forward_chunk(
        self,
        audio_16khz,
    ):
        was_silence = self.in_speech_cooldown == 0
        was_speech = self.in_speech_cooldown > 0

        speech_prob, self.state = self.model(audio_16khz, self.state)
        trigger = speech_prob >= self.start_threshold
        decay = speech_prob < self.end_threshold

        self.in_speech_cooldown[trigger] = self.min_silence_samples
        self.in_speech_cooldown[decay] = torch.clamp(
            self.in_speech_cooldown[decay] - self.n_samples, min=0
        )

        is_speech = self.in_speech_cooldown > 0
        is_starting = is_speech & was_silence
        is_ending = ~is_speech & was_speech
        is_not_speech = ~is_speech & ~is_ending & ~is_starting

        return is_speech, is_not_speech, is_starting, is_ending, speech_prob

    def initialize(self, batch_size: int, device: str):
        self.state = self.model.init_state(batch_size, device)
        self.in_speech_cooldown = torch.zeros(batch_size, device=device).long()
