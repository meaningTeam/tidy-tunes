import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceActivityDetector(nn.Module):
    SAMPLING_RATE = 16000
    WINDOW_SAMPLES = 512  # 32ms at 16kHz

    def __init__(
        self,
        model,
        min_silence_chunks: int = 10,
        start_threshold: float = 0.7,
        end_threshold: float = 0.2,
    ):
        """
        Voice Activity Detector using Silero VAD.

        Args:
            model: Silero VAD model loaded via silero_vad.load_silero_vad().
            min_silence_chunks: Minimum consecutive silence frames (32ms each)
                before ending a speech segment (default: 10 = 320ms).
            start_threshold: Probability threshold to start speech detection.
            end_threshold: Probability threshold to stop speech detection.
        """
        super().__init__()
        assert start_threshold >= end_threshold

        self.model = model
        self.frame_shift = self.WINDOW_SAMPLES / self.SAMPLING_RATE
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.n_samples = self.WINDOW_SAMPLES
        self.min_silence_samples = min_silence_chunks * self.n_samples

    @property
    def sampling_rate(self):
        return self.SAMPLING_RATE

    @torch.no_grad()
    def forward(self, audio_16khz):
        """
        Processes audio signals to detect voice activity.

        Args:
            audio_16khz (B, T): Input audio waveforms at 16kHz.

        Returns:
            Binary mask (B, L) indicating speech presence per frame.
        """
        audio_16khz = torch.atleast_2d(audio_16khz)
        audio_16khz = F.pad(audio_16khz, (0, self.n_samples - 1))
        B, T = audio_16khz.shape

        all_masks = []
        for i in range(B):
            wav = audio_16khz[i]
            self.model.reset_states()

            max_length = (len(wav) // self.n_samples) * self.n_samples
            probs = []
            for j in range(0, max_length, self.n_samples):
                chunk = wav[j : j + self.n_samples]
                prob = self.model(chunk, self.SAMPLING_RATE)
                probs.append(prob)

            if not probs:
                all_masks.append(
                    torch.zeros(0, device=audio_16khz.device, dtype=torch.bool)
                )
                continue

            probs_t = torch.stack(probs)
            mask = self._apply_hysteresis(probs_t, audio_16khz.device)
            all_masks.append(mask)

        max_len = max(len(m) for m in all_masks) if all_masks else 0
        result = torch.zeros(B, max_len, device=audio_16khz.device, dtype=torch.bool)
        for i, m in enumerate(all_masks):
            result[i, : len(m)] = m

        return result

    def _apply_hysteresis(self, probs, device):
        mask = torch.zeros(len(probs), device=device, dtype=torch.bool)
        cooldown = 0
        for j in range(len(probs)):
            p = probs[j].item()
            if p >= self.start_threshold:
                cooldown = self.min_silence_samples
            elif p < self.end_threshold:
                cooldown = max(cooldown - self.n_samples, 0)
            mask[j] = cooldown > 0
        return mask
