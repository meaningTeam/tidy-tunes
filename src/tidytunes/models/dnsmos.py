import math

import numpy as np
import onnxruntime as ort
import torch
from torchaudio.transforms import MelSpectrogram


class DNSMOSPredictor(torch.nn.Module):

    def __init__(
        self,
        base_model: ort.InferenceSession,
        p808_model: ort.InferenceSession,
        model_device: str,
        personalized: bool,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        input_duration: float = 9.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.p808_model = p808_model
        self.model_device = model_device
        self.personalized = personalized
        self.frame_size = frame_size
        self.sampling_rate = sr
        self.feature_extractor = MelSpectrogram(
            sample_rate=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )
        self.input_duration = input_duration

    @staticmethod
    def power_to_db(
        x: torch.Tensor,
        amin: float = 1e-10,
        top_db: float = 80.0,
    ):
        ref_value = x.max().abs()
        log_spec = 10.0 * torch.clamp(x, min=amin).log10()
        log_spec -= 10.0 * math.log10(max(amin, ref_value))
        log_spec = torch.clamp(log_spec, min=log_spec.max() - top_db)
        return log_spec

    def extract_features(
        self,
        audio: torch.Tensor,
    ):
        features = self.feature_extractor(audio)
        features = (self.power_to_db(features) + 40) / 40
        return features

    def get_polyfit_val(self, mos):

        if self.personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        mos[..., 1] = p_sig(mos[..., 1])
        mos[..., 2] = p_bak(mos[..., 2])
        mos[..., 3] = p_ovr(mos[..., 3])
        return mos

    def forward(
        self,
        audio_16khz: torch.Tensor,
        lens: torch.Tensor,
    ):
        """
        Calculate Deep Noise Suppression performance evaluation based on Mean Opinion Score

        Args:
            audio_16khz (B, T): Batch of waveforms to evaluate. Must be sampled at 16 kHz.
            lens (B,): Lenghts of valid audio samples in `audio_16khz`.

        Returns:
            tuple(p808_mos, mos_sig, mos_bak, mos_ovr) (B,): Tuple of tensors of DNSMOS values for each batch item.
        """

        B, _ = audio_16khz.shape

        audio_16khz = audio_16khz.to(self.model_device)
        lens = lens.to(self.model_device)

        device = audio_16khz.device
        idx = 0 if device.index is None else device.index

        base_binding = self.base_model.io_binding()
        p808_binding = self.p808_model.io_binding()

        output = torch.empty((B, 3), dtype=torch.float32, device=device).contiguous()
        base_binding.bind_output(
            name="Identity:0",
            device_type=device.type,
            device_id=idx,
            element_type=np.float32,
            shape=(B, 3),
            buffer_ptr=output.data_ptr(),
        )

        p808_output = torch.empty(
            (B, 1), dtype=torch.float32, device=device
        ).contiguous()
        p808_binding.bind_output(
            name="Identity:0",
            device_type=device.type,
            device_id=idx,
            element_type=np.float32,
            shape=(B, 1),
            buffer_ptr=p808_output.data_ptr(),
        )

        durations = lens / self.sampling_rate
        pad_len = ((self.input_duration / durations).ceil() * lens).max()
        sample_idx = (
            torch.arange(pad_len, device=device).unsqueeze(0).expand(lens.shape[0], -1)
        )
        sample_idx = torch.remainder(sample_idx, lens.unsqueeze(-1)).long()
        audio_padded = torch.gather(audio_16khz, 1, sample_idx)

        moss = []

        # NOTE: (@tomiinek) This is kind of weird but it is how DNSMOS works. The hop is 1s and
        # the MOS is evlauated on possibly multiple chunks of the audio. Depends on its length.

        num_seconds = (
            int(
                np.floor(audio_padded.shape[-1] / self.sampling_rate)
                - self.input_duration
            )
            + 1
        )
        second_samples = int(1.0 * self.sampling_rate)
        for start_second in range(num_seconds):

            s = int(start_second * second_samples)
            e = int((start_second + self.input_duration) * second_samples)
            audio_seg = audio_padded[..., s:e]

            if audio_seg.shape[-1] < self.input_duration * second_samples:
                break

            shape = audio_seg.shape
            audio_seg = audio_seg.reshape((-1, shape[-1]))
            mel_seg = self.extract_features(audio_seg[..., :-160]).transpose(1, 2)

            base_binding.bind_input(
                name="input_1",
                device_type=device.type,
                device_id=idx,
                element_type=np.float32,
                shape=tuple(audio_seg.shape),
                buffer_ptr=audio_seg.data_ptr(),
            )
            p808_binding.bind_input(
                name="input_1",
                device_type=device.type,
                device_id=idx,
                element_type=np.float32,
                shape=tuple(mel_seg.shape),
                buffer_ptr=mel_seg.data_ptr(),
            )

            self.p808_model.run_with_iobinding(p808_binding)
            self.base_model.run_with_iobinding(base_binding)

            mos_np = torch.cat([p808_output, output], dim=-1).cpu().numpy()
            mos_np = self.get_polyfit_val(mos_np)

            mos_np = mos_np.reshape(shape[:-1] + (4,))
            moss.append(mos_np)

        mos = torch.from_numpy(np.mean(np.stack(moss, axis=-1), axis=-1))
        return mos.unbind(dim=-1)
