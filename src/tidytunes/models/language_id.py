import json
from pathlib import Path

import torch

from tidytunes.models.external.speechbrain_ecapa_tdnn import (
    ECAPA_TDNN,
    Classifier,
    Fbank,
    InputNormalization,
)
from tidytunes.utils.trace import TraceMixin


class SpokenLanguageIdentificationModel(torch.nn.Module, TraceMixin):

    def __init__(self, label_to_language_map_path: Path):
        super().__init__()
        self.compute_features = Fbank()
        self.mean_var_norm = InputNormalization()
        self.embedding_model = ECAPA_TDNN()
        self.classifier = Classifier()
        with open(label_to_language_map_path, "r") as f:
            self.ind2lab = {int(k): v for k, v in json.load(f).items()}
        self.lab2ind = {lang: ind for ind, lang in self.ind2lab.items()}

    @classmethod
    def from_files(
        cls,
        classifier_path: Path,
        embedding_model_path: Path,
        normalizer_path: Path,
        label_to_language_map_path: Path,
        map_location: str = "cpu",
    ):
        model = cls(label_to_language_map_path)
        model.mean_var_norm.load(normalizer_path, map_location=map_location)
        model.embedding_model.load_state_dict(
            torch.load(
                embedding_model_path, map_location=map_location, weights_only=True
            )
        )
        model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=map_location, weights_only=True)
        )
        return model

    def encode_batch(
        self, audio_16khz: torch.Tensor, audio_16khz_lens: torch.Tensor | None = None
    ):
        """
        Encodes the input audio into a single vector embedding.

        Args:
            audio_16khz (B, T): Batch of waveforms at 16 kHz.
            audio_16khz_lens (B,): Lengths of the waveforms.

        Returns:
            embeddings (B, L, D): Waveform embeddings with frame rate of internal mel extractor.
        """
        if len(audio_16khz.shape) == 1:
            audio_16khz = audio_16khz.unsqueeze(0)

        if audio_16khz_lens is None:
            audio_16khz_lens = torch.ones(
                audio_16khz.shape[0], device=audio_16khz.device
            )
        else:
            audio_16khz_lens = audio_16khz_lens.float()
            audio_16khz_lens /= audio_16khz_lens.max()

        audio_16khz = audio_16khz.float()

        feats = self.compute_features(audio_16khz)
        feats = self.mean_var_norm(feats, audio_16khz_lens)
        embeddings = self.embedding_model(feats, audio_16khz_lens)

        return embeddings

    def forward(
        self, audio_16khz: torch.Tensor, audio_16khz_lens: torch.Tensor | None = None
    ):
        """
        Performs language classification of input waveforms.

        Args:
            audio_16khz (B, T): Batch of waveforms at 16 kHz.
            audio_16khz_lens (B,): Lengths of the waveforms.

        Returns:
            out_prob (B, C): The probabilities of each class.
            score (B,): It is the probability for the best class.
            index (B,): The indexes of the best class.
        """
        emb = self.encode_batch(audio_16khz, audio_16khz_lens)
        out_prob = self.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        return out_prob, score, index

    def dummy_inputs(
        self,
        batch: int = 2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        return (
            torch.randn(batch, 16000 * 5, device=device, dtype=dtype),
            torch.tensor(batch * [16000 * 5], device=device).long(),
        )
