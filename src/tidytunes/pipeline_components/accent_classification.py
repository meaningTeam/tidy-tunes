from functools import lru_cache

import torch

from tidytunes.utils import Audio, batched


@batched(batch_size=128, batch_duration=1280.0)
def get_accent_probabilities(
    audio: list[Audio],
    accent_label: str,
    model_id: str = "badrex/mms-300m-arabic-dialect-identifier",
    device: str = "cpu",
) -> list[float] | list[tuple[float, dict]]:
    """
    Compute the probability of a given accent/dialect being spoken in the audio.

    Args:
        audio (list[Audio]): List of Audio objects to analyze.
        accent_label (str): The target accent/dialect label (e.g. "MSA", "Egyptian",
            "Gulf", "Levantine", "Maghrebi"). Must match one of the model's class labels.
        model_id (str): HuggingFace model identifier for the accent classifier.
        device (str): The device to run the model on (default: "cpu").

    Returns:
        list[float]: Per-audio probabilities for the specified accent label. When used
            with ``annotate``, returns list[tuple[float, dict]] where each dict contains
            the full probability distribution over all accent labels.
    """
    model, feature_extractor, label2id = load_accent_classifier(model_id, device)

    if accent_label not in label2id:
        available = ", ".join(sorted(label2id.keys()))
        raise ValueError(
            f"Unknown accent label '{accent_label}'. Available labels: {available}"
        )

    target_idx = label2id[accent_label]
    id2label = model.config.id2label
    raw_waveforms = [a.resample(16000).data.numpy() for a in audio]

    inputs = feature_extractor(
        raw_waveforms,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    return [
        (
            p[target_idx].item(),
            {id2label[i]: p[i].item() for i in range(len(p))},
        )
        for p in probs
    ]


@lru_cache(maxsize=1)
def load_accent_classifier(model_id: str, device: str = "cpu"):
    """
    Loads and caches the HuggingFace accent/dialect classification model.

    Args:
        model_id (str): HuggingFace model identifier.
        device (str): The device to load the model on (default: "cpu").

    Returns:
        Tuple of (model, feature_extractor, label2id mapping).
    """
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
    model = model.eval().to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    label2id = {label: idx for idx, label in model.config.id2label.items()}

    return model, feature_extractor, label2id
