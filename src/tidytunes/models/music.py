import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


class MusicDetectionModel(nn.Module):
    """
    Frame-level music detection model based on wav2vec2-bert.

    Adds a classification head on top of the encoder outputs for
    binary classification (music present / not present) per frame.
    """

    def __init__(self, model_name: str = "facebook/w2v-bert-2.0"):
        super().__init__()
        self.encoder = Wav2Vec2BertModel.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = 16000
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        audio_16khz: torch.Tensor,
        audio_16khz_lens: torch.Tensor,
    ) -> torch.Tensor:

        inputs = self.feature_extractor(
            [a[:al].cpu().numpy() for a, al in zip(audio_16khz, audio_16khz_lens)],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.encoder(
            input_features=inputs.input_features.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
        )
        hidden_states = outputs.last_hidden_state  # [B, T', H]
        logits = self.classifier(hidden_states).squeeze(-1)  # [B, T']
        probs = torch.sigmoid(logits)

        lens = (audio_16khz_lens - 400) // 320 + 1

        return probs, lens

    @classmethod
    def from_files(cls, model_weights_path: str) -> "MusicDetectionModel":
        model = cls()
        model.load_state_dict(torch.load(model_weights_path))
        return model
